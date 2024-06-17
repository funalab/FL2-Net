# -*- coding: utf-8 -*-

import json
import shutil
import time
import numpy as np
import copy
import os
import sys
import re
import shutil
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

sys.path.append(os.getcwd())
from src.test import Tester
from src.utils.misc import load_yaml
from src.utils.utils import createOpbase
from src.evaluation import NSNEvaluator, NDNEvaluator, ClassificationEvaluator
from src.datasets.dataset import PreprocessedDataset
from src.modeling import get_model
sys.setrecursionlimit(200000)


if __name__ == '__main__':

    start_time = time.time()
    ap = ArgumentParser(description='python validation.py')
    ap.add_argument('--test_conf', nargs='?', default='config/test/test.yaml', help='Specify configuration file for test time')
    ap.add_argument('--model_conf', nargs='?', default='config/model/Watershed/watershed_v2_gru.yaml', help='Specify configuration file for modeling')
    ap.add_argument('--output_dir', '-o', nargs='?', default='results/val', help='Specify output files directory for create figures')
    ap.add_argument('--model_dir', '-m', nargs='?', default='results/trained_models', help='Specify direcotry for loading files of Learned Model')
    ap.add_argument('--iter_eval', '-ie', type=int, default=0, help='start evaluation')
    ap.add_argument('--del_model', action='store_true', help='if true, delete low performance model')

    args = ap.parse_args()
    argvs = sys.argv

    cfg = load_yaml(args.model_conf, args.test_conf)

    ''' create output directory '''
    opbase = createOpbase(args.output_dir)
    shutil.copyfile(args.model_conf, "{}/model_config.yaml".format(opbase))
    shutil.copyfile(args.test_conf, "{}/test_config.yaml".format(opbase))

    ''' Data Iterator '''
    print('Loading datasets...')
    test_dataset = PreprocessedDataset(cfg, train=False)
    print('validation dataset.size = {}\n'.format(len(test_dataset)))
    data_iterator = DataLoader(
        dataset=test_dataset,
        batch_size=int(cfg.DATASETS.BATCH_SIZE),
        shuffle=False
    )
    #data_iterator = get_dataloader(cfg, train=False)

    ''' model list '''
    mlist = os.listdir(args.model_dir)
    mlist = np.sort(mlist)

    ''' Model '''
    model = get_model(cfg)

    ''' create evaluator '''
    if cfg.TASK == 'segmentation':
        criteria = ['IoU', 'MUCov', 'SEG']
        evaluator = NSNEvaluator(criteria=criteria, connectivity=1)
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
    validator = Tester(evaluator, criteria, opbase, cfg, save_pred=False)

    max_val = 0
    metrics = {}

    for ml in mlist:
        ''' Load model '''
        model_name = os.path.join(args.model_dir, ml)
        print('Load model from', model_name)
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to('cuda')

        model_iter = int(re.sub(r"\D", "", ml))
        ### skip ###
        if model_iter < args.iter_eval:
            continue

        evals_avg, evals_all = validator.test(dataloader=data_iterator, model=model)
        metrics[str(model_iter)] = evals_avg

        with open(os.path.join(opbase, 'result.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(opbase, 'result.txt'), 'a') as f:
            f.write('iteration: {}\n'.format(str(model_iter)))
            for cri, vals in evals_all.items():
                for key, val in vals.items():
                    f.write('{}\n'.format(key))
                    f.write('{}: {}\n'.format(cri, val))
            f.write('\n')

        if args.del_model:
            if max_val <= evals[indicator]:
                max_val = evals[indicator]
                shutil.copy2(model_name, opbase + '/best_{}_model.npz'.format(indicator))
            os.remove(ml)

    end_time = time.time()
    etime = end_time - start_time
    print('Elapsed time is (sec) {}'.format(etime))
    with open(opbase + '/result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(etime))
    print('{} Test Completed Process!'.format(cfg.TASK))
