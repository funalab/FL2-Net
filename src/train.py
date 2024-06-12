# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
import argparse
import configparser
import yaml
import json
import shutil
import tifffile

import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.utils.utils import createOpbase, get_optimizer
from src.utils.misc import load_yaml
from src.evaluation import NDNEvaluator, NSNEvaluator, ClassificationEvaluator
from src.datasets.dataset import PreprocessedDataset
from src.modeling import get_model

seed = 109
#seed = 110
#seed = 111


class Trainer():

    def __init__(self, evaluator, criteria, indicator, opbase, cfg):
        self.opbase = opbase
        self.criteria = criteria
        self.evaluator = evaluator
        self.indicator = indicator

        # load from config file
        self.epoch = cfg.RUNTIME.TRAIN_EPOCH
        self.iter_interval = cfg.LOG.EVALUATION_INTERVAL
        self.iter_eval = cfg.LOG.EVALUATION_START
        self.half_precision = bool(cfg.RUNTIME.HALF_PRECISION)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.half_precision)
        
        # variable for logging
        self.iteration = 1

        if cfg.MODEL.META_ARCHITECTURE == 'DeepWatershed':
            self.loss_sum_dict = {"loss_mask":0.0, "loss_center":0.0, "loss_sum":0.0}

        elif cfg.MODEL.META_ARCHITECTURE == 'DeepWatershedV2':
            self.loss_sum_dict = {"loss_mask":0.0, "loss_center":0.0, "loss_dist":0.0, "loss_sum":0.0}
        
        elif cfg.MODEL.META_ARCHITECTURE == 'EmbedSeg':
            self.loss_sum_dict = {"loss_inst":0.0, "loss_var":0.0, "loss_seed":0.0, "loss_sum":0.0}
        
        elif cfg.MODEL.META_ARCHITECTURE == 'StarDist':
            self.loss_sum_dict = {"loss_dist":0.0, "loss_obj":0.0, "loss_sum":0.0}

        else:
            self.loss_sum_dict = {"loss_sum": 0.0}
        self.evals_sum = dict(zip(criteria, [0]*len(criteria)))
        self.evals_iter = {}
    
    def training(self, model, train_iter, optimizer):

        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('N_train: {}\n'.format(train_iter.dataset.__len__()))
        
        os.makedirs(os.path.join(self.opbase, 'checkpoints'), exist_ok=True)

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            
            # start epoch
            model.train()
            train_loss = self._trainer(model, train_iter, optimizer)

            print('train mean loss={}'.format(train_loss))
            with open(self.opbase + '/result.txt', 'a') as f:
                f.write('========================================\n')
                f.write('[epoch' + str(epoch) + ']\n')
                f.write('train mean loss={}\n\n'.format(train_loss))
            
            # save checkpoint
            torch.save({
                'epoch': epoch,
                #'data_state_dict': train_iter.get_state(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(self.opbase, 'checkpoints', 'checkpoint_{}ep.pth'.format(epoch)))

    
    def _trainer(self, model, dataset_iter, optimizer):
        
        loss_list = []

        for batch in dataset_iter:

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.half_precision):
                outputs = model(batch, mode='train', inference=(self.iteration >= self.iter_eval))
                loss = outputs['loss']
            self.scaler.scale(loss['loss_sum']).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            for k, v in loss.items():
                loss[k] = float(v.to(torch.device('cpu')))
                self.loss_sum_dict[k] += loss[k]
            loss_list.append(float(loss['loss_sum']))
            del loss

            evals_avg = {}
            if self.iteration > self.iter_eval:
                # evaluation
                target = batch['instances'].to(torch.device('cpu')).numpy()[0]
                pred = outputs['instances'].to(torch.device('cpu')).detach().numpy()[0]

                evals = self.evaluator.evaluation(pred, target)
                for cri in self.criteria:
                    self.evals_sum[cri] += evals[cri]
                    evals_avg[cri] = self.evals_sum[cri] / self.iter_interval
            del outputs
            
            if self.iteration % self.iter_interval == 0:
                evals_avg['loss'] = {}
                for k, v in self.loss_sum_dict.items():
                    evals_avg['loss'][k] = v / self.iter_interval
                self.evals_iter[self.iteration] = evals_avg
                with open(os.path.join(self.opbase, 'log_iter.json'), 'w') as f:
                    json.dump(self.evals_iter, f, indent=4)
                self.loss_sum_dict = dict(zip(self.loss_sum_dict.keys(), [0.0]*len(self.loss_sum_dict)))
                self.evals_sum = dict(zip(self.criteria, [0]*len(self.criteria)))

                # save model parameters
                model_name = 'model_{0:06d}.pth'.format(self.iteration)
                torch.save({
                    'iteration': self.iteration,
                    'model_state_dict': model.state_dict(),
                }, os.path.join(self.opbase, 'trained_models',  model_name))
                model.to('cuda')

            self.iteration += 1

        return float(abs(np.mean(loss_list)))


def main():

    """ Implementation of Transformer-based 3D instance Segmentation Network """
    start_time = time.time()

    ''' ConfigParser '''
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-r", "--runtime_conf", help="Specify config file for runtime", metavar="FILE_PATH")
    conf_parser.add_argument("-m", "--model_conf", help="Specify config file for model", metavar="FILE_PATH")
    conf_parser.add_argument("-o", "--output_dir", help="Specify output directory name", nargs='?', default='results/mask3dformer/train',)
    args = conf_parser.parse_args()
    argvs = sys.argv

    cfg = load_yaml(args.model_conf, args.runtime_conf)

    ''' create output directory '''
    opbase = createOpbase(args.output_dir)
    os.makedirs(os.path.join(opbase, 'trained_models'), exist_ok=True)
    shutil.copyfile(args.model_conf, "{}/model_config.yaml".format(opbase))
    shutil.copyfile(args.runtime_conf, "{}/runtime_config.yaml".format(opbase))
    #print_args(dataset_args, model_args, runtime_args)
    with open(opbase + '/result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ''' Data Iterator '''
    print('Loading datasets...')
    #train_iterator = get_dataloader(cfg, train=True)
    train_dataset = PreprocessedDataset(cfg, train=True)
    print('train_dataset.size = {}\n'.format(len(train_dataset)))
    train_iterator = DataLoader(
        dataset=train_dataset,
        batch_size=int(cfg.DATASETS.BATCH_SIZE),
        shuffle=True
    )


    ''' Model '''
    print('Initializing models...')
    model = get_model(cfg)
    
    ''' Optimizer '''
    print('Initializing optimizer...')
    optimizer = get_optimizer(cfg, model)
    
    ''' loading checkpoint '''
    if cfg.MODEL.CHECKPOINT_PATH:
        print('Load checkpoint from', cfg.MODEL.CHECKPOINT_PATH)
        checkpoint = torch.load(cfg.MODEL.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to('cuda')

    ''' create evaluator '''
    if cfg.TASK == 'segmentation' or cfg.TASK =='tracking':
        criteria = ['accuracy', 'recall', 'precision', 'specificity', 'F-measure', 'IoU']
        evaluator = NSNEvaluator(criteria=criteria, connectivity=1)
        indicator = ['IoU']

    elif cfg.TASK == 'detection':
        criteria = ['recall', 'precision', 'F-measure', 'IoU']
        evaluator = NDNEvaluator(criteria=criteria, connectivity=1, radius=10, delv=0)
        indicator = ['F-measure']

    else:
        print('Warning: select segmentation or detection')
        sys.exit()

    ''' Training Phase '''
    trainer = Trainer(evaluator, criteria, indicator, opbase, cfg)
    trainer.training(model, train_iterator, optimizer)
    
    end_time = time.time()
    process_time = end_time - start_time
    print('Elapsed time is (sec) {}'.format(process_time))
    with open(opbase + '/result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(process_time))

if __name__ == '__main__':
    main()
