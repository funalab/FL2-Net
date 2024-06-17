# -*- coding: utf-8 -*-
# ToDo: for multiple batch
import csv
import sys
import time
import random
import copy
import math
import os
import json
import shutil
import numpy as np
import os.path as pt
from skimage import io
from skimage import transform as tr
from skimage import morphology as mor
from argparse import ArgumentParser
from concurrent.futures.process import ProcessPoolExecutor
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.getcwd())
from src.utils.utils import mirror_extension_image, createOpbase, labeling
from src.utils.misc import load_yaml
from src.evaluation import NDNEvaluator, NSNEvaluator, ClassificationEvaluator
from src.datasets.dataset import PreprocessedDataset
from src.modeling import get_model


class Tester():
    def __init__(self, evaluator, criteria, opbase, cfg, save_pred=True):
        self.opbase = opbase
        self.evaluator = evaluator
        self.criteria = criteria
        self.opbase = opbase
        self.task = cfg.TASK
        if cfg.DATASETS.INPUT_SIZE:
            self.patchsize = cfg.DATASETS.INPUT_SIZE
            self.stride = (int(self.patchsize[0]//2), int(self.patchsize[1]//2), int(self.patchsize[2]//2))
            self.stitch = True
        else:
            self.stitch = False
        self.resolution = cfg.DATASETS.RESOLUTION
        self.scaling = cfg.DATASETS.PREPROCESS.SCALING
        self.half_precision = cfg.TEST_TIME.HALF_PRECISION
        self.save_pred = save_pred
        if save_pred:
            segbase = opbase + '/Predictions'
            if not (pt.exists(segbase)):
                os.mkdir(segbase)

    def test(self, dataloader, model):
        model.eval()

        evals_sum = dict(zip(self.criteria, [0]*len(self.criteria)))
        results_all = {cri:{} for cri in self.criteria}

        for batch in dataloader:
            f_name = batch['filename'][0]
            emb_name = os.path.dirname(f_name).replace('/', '_')
            tp_name = os.path.basename(f_name)[:os.path.basename(f_name).rfind('.')]

            if self.task == 'segmentation' or self.task == "detection":
                # segmentation prediction
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.half_precision), torch.no_grad():
                    if self.stitch:
                        output = model(batch, mode='test', inference=True, patchsize=self.patchsize)
                    else:
                        output = model(batch, mode='test', inference=True)

                pred = output['instances'].detach().to(torch.device('cpu')).numpy()[0]
                pred = tr.resize(pred, batch['image_size'][0], order=0, preserve_range=True, anti_aliasing=False)
                target = batch['instances'].to(torch.device('cpu')).numpy()[0] if ('instances' in batch) else None

                if self.save_pred:
                    if not (pt.exists(os.path.join(self.opbase, 'Predictions', emb_name))):
                        os.mkdir(os.path.join(self.opbase, 'Predictions', emb_name))
                    filename = os.path.join(self.opbase, 'Predictions', emb_name, tp_name + '.tif')
                    io.imsave(filename, pred.astype(np.uint16))

            elif self.task == 'classification':
                # classification prediction
                pred = self._classification(batch)
                target = batch['dev_class'].to(torch.device('cpu')).numpy()[0]

                if self.save_pred:
                    filename = os.path.join(self.opbase, 'Predictions', emb_name, tp_name + '.tif')
                    with open(os.path.join(self.opbase, 'Predictions', emb_name + '.txt'), 'a') as f:
                        f.write("{} : {}".format(tp, pred))
            
            else:
                print('Specify task [classification or segmentation]')
            
            if self.evaluator != None:
                evals = self.evaluator.evaluation(pred, target)
                for cri in self.criteria:
                    key = f_name[:f_name.rfind('.')]
                    results_all[cri][key] = evals[cri]
                    evals_sum[cri] += evals[cri]

        return {k: v / dataloader.dataset.__len__() for k, v in evals_sum.items()}, results_all
    
    
if __name__ == '__main__':

    start_time = time.time()
    ap = ArgumentParser(description='python test.py')
    ap.add_argument('--test_conf', nargs='?', default='', help='Specify configuration file for test time')
    ap.add_argument('--model_conf', nargs='?', default='confs/model/Watershed/watershed_v2_gru.yaml', help='Specify configuration file for modeling')
    ap.add_argument('--model', '-m', nargs='?', default='models/deepws_gru.pth', help='Specify loading file path of learned Model')
    ap.add_argument("--output_dir", "-o", nargs='?', default='results/train', help="Specify output directory name")
    ap.add_argument('--save_img', action='store_true', help='if true, save output images')
    ap.add_argument('--evaluation', action='store_true', help='if true, evaluate performance')
    
    args = ap.parse_args()
    argvs = sys.argv

    cfg = load_yaml(args.model_conf, args.test_conf)

    ''' create output directory '''
    opbase = createOpbase(args.output_dir)
    shutil.copyfile(args.model_conf, "{}/model_config.yaml".format(opbase))
    shutil.copyfile(args.test_conf, "{}/test_config.yaml".format(opbase))
    with open(opbase + '/result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    # Load Model
    ''' Data Iterator '''
    print('Loading datasets...')
    #data_iterator = get_dataloader(cfg, train=False)
    test_dataset = PreprocessedDataset(cfg, train=False)
    print('test_dataset.size = {}\n'.format(len(test_dataset)))
    data_iterator = DataLoader(
        dataset=test_dataset,
        batch_size=int(cfg.DATASETS.BATCH_SIZE),
        shuffle=False
    )

    ''' Model '''
    print('Load Model from', args.model)
    model = get_model(cfg)
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')

    ''' create evaluator '''
    if not args.evaluation:
        evaluator = None
        criteria = []
        
    elif cfg.TASK == 'segmentation':
        criteria = ['IoU', 'MUCov', 'SEG']
        evaluator = NSNEvaluator(criteria=criteria, connectivity=0)
        indicator = 'IoU'

    elif cfg.TASK == 'detection':
        criteria = ['recall', 'precision', 'F-measure', 'IoU']
        evaluator = NDNEvaluator(criteria=criteria, connectivity=1, radius=10, delv=0)
        indicator = 'F-measure'
        
    elif cfg.TASK == 'classification':
        criteria = ['accuracy', ]
        evaluator = ClassificationEvaluator(criteria=criteria)
        indicator = 'accuracy'
    else:
        print('Warning: select segmentation or detection or classification')
        sys.exit()

    # Prediction Phase
    tester = Tester(evaluator, criteria, opbase, cfg, save_pred=args.save_img)
    evals_avg, evals_all = tester.test(dataloader=data_iterator, model=model)

    with open(opbase + '/result_avg.json', 'a') as f:
        json.dump(evals_avg, f, indent=4)

    with open(opbase + '/results.json', 'a') as f:
        json.dump(evals_all, f, indent=4)

    end_time = time.time()
    etime = end_time - start_time
    print('Elapsed time is (sec) {}'.format(etime))
    with open(opbase + '/result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(etime))
    print('{} Test Completed Process!'.format(cfg.TASK))
