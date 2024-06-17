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
import tifffile
from skimage import io
from skimage import transform as tr
from skimage import morphology as mor
from scipy import ndimage
from skimage.measure import label
from skimage import morphology
from skimage.segmentation import watershed
from argparse import ArgumentParser
from concurrent.futures.process import ProcessPoolExecutor

sys.path.append(os.getcwd())
from src.utils.utils import createOpbase
from src.utils.misc import load_yaml
from src.evaluation import NDNEvaluator, NSNEvaluator, ClassificationEvaluator


if __name__ == '__main__':

    start_time = time.time()
    ap = ArgumentParser(description='python test.py')
    ap.add_argument('--dataset', '-l', nargs='?', default='datasets/split_list_411/test.txt', help='Specify split list for test data')
    ap.add_argument('--nsn_dir', nargs='?', default='results/nsn', help='Specify input directory for semantic segmentation images')
    ap.add_argument('--ndn_dir', nargs='?', default='results/ndn', help='Specify input directory for detection images')
    ap.add_argument('--gt_dir', nargs='?', default=None, help='Specify ground truth directory for instance segmentation images')
    ap.add_argument("--out_dir", "-o", nargs='?', default='results/qcanet/test_instance', help="Specify output directory name")
    ap.add_argument('--evaluation', action='store_true', help='if true, evaluate performance')
    
    args = ap.parse_args()
    argvs = sys.argv
    
    ''' create output directory '''
    opbase = createOpbase(args.out_dir)
    os.mkdir(opbase + '/Predictions')

    ''' create evaluator '''
    criteria = ['IoU', 'MUCov', 'SEG']
    if args.evaluation:
        evaluator = NSNEvaluator(criteria=criteria, connectivity=0)
    else:
        evaluator = None
    indicator = 'IoU'


    # Load Model
    ''' Data Iterator '''
    print('Loading datasets...')
    with open(args.dataset, 'r') as f:
        datasets = f.read().split()

    evals_sum = dict(zip(criteria, [0]*len(criteria)))
    evals_all = {cri:{} for cri in criteria}

    for f_name in datasets:
        emb_name = os.path.dirname(f_name)
        tp_name = os.path.basename(f_name)[:os.path.basename(f_name).rfind('.')]
        out_name = os.path.join(opbase, 'Predictions', emb_name.replace('/', '_'), tp_name+'.tif')
        if not (pt.exists(os.path.join(opbase, 'Predictions', emb_name.replace('/', '_')))):
            os.mkdir(os.path.join(opbase, 'Predictions', emb_name.replace('/', '_')))
        ### loading output image ###
        ndn = tifffile.imread(os.path.join(args.ndn_dir, emb_name.replace('/', '_'), tp_name+'.tif'))
        ndn = label((ndn > 0) * 1, connectivity=3)
        nsn = tifffile.imread(os.path.join(args.nsn_dir, emb_name.replace('/', '_'), tp_name+'.tif'))
        nsn = (nsn > 0) * 1
        ### watershed ###
        dist = ndimage.distance_transform_edt(nsn)
        instance = watershed(-dist, ndn, mask=nsn)
        tifffile.imsave(out_name, instance.astype(np.uint16))
        ### evaluation ###
        if evaluator != None:
            ### loading GT image ###
            gt = tifffile.imread(os.path.join(args.gt_dir, emb_name, tp_name+'.tif'))
            evals = evaluator.evaluation(instance, gt)
            for cri in criteria:
                key = f_name[:f_name.rfind('.')]
                evals_all[cri][key] = evals[cri]
                evals_sum[cri] += evals[cri]

    evals_avg = {k: v / len(datasets) for k, v in evals_sum.items()}

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
