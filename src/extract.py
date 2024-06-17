#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
sys.path.append(os.getcwd())
import json
import numpy as np
import pytz
from glob import glob
from argparse import ArgumentParser
from datetime import datetime
from skimage import io
from skimage import morphology, measure, transform
from skimage.measure import regionprops
from scipy import ndimage
import argparse
import configparser

from src.utils.utils import createOpbase


def main(argv=None):
    starttime = time.time()

    ap = ArgumentParser(description='python extract.py')
    ap.add_argument('--indir', '-i', nargs='?', default='results/pred', help='Specify predicted images')
    ap.add_argument('--output', '-o', nargs='?', default='results/feat', help='Specify output file directory')
    ap.add_argument('--split_list', '-l', nargs='?', default=None, help='Specify split list file')
    ap.add_argument('--max_time', '-t', type=int, default=506, help='Specify max timepoint')
    ap.add_argument('--time_resolution', '-tr', type=int, default=10, help='Specify time resolution (default=10)')
    ap.add_argument('--resolution_x', '-x', type=float, default=0.8, help='Specify microscope resolution of x axis (default=1.0)')
    ap.add_argument('--resolution_y', '-y', type=float, default=0.8, help='Specify microscope resolution of y axis (default=1.0)')
    ap.add_argument('--resolution_z', '-z', type=float, default=2.0, help='Specify microscope resolution of z axis (default=2.5)')

    args = ap.parse_args()
    argvs = sys.argv

    max_tp = args.max_time
    tp = ['{0:03d}'.format(i) for i in range(1, max_tp+1)]

    # Resolution
    spatial_resolution = np.array([args.resolution_z, args.resolution_y, args.resolution_x])
    voxel_convert_scale = spatial_resolution[0] * spatial_resolution[1] * spatial_resolution[2]
    # spatial_resolution /= np.min(spatial_resolution)
    time_resolution = args.time_resolution

    # Make Directory
    opbase = createOpbase(args.output)

    # Selection Criteria
    criteria_list = [
        'number', 'volume_sum', 'volume_mean', 'volume_sd',
        'surface_sum', 'surface_mean', 'surface_sd',
        'aspect_ratio_mean', 'aspect_ratio_sd',
        'solidity_mean', 'solidity_sd', 'centroid_mean', 'centroid_sd'
    ]

    # Make log
    if args.split_list == None:
        path_directory = np.sort(os.listdir(args.indir))
    else:
        with open(args.split_list, 'r') as f:
            path_directory = np.sort(f.read().split())
    print(path_directory)
    print(criteria_list)
    with open(opbase + '/log.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n\n')
        f.write('target list: {}\n'.format(path_directory))
        f.write('criteria list: {}\n'.format(criteria_list))

    # Parameter of image processing
    kernel = np.ones((3,3,3),np.uint8)

    for pd in path_directory:
        with open(opbase + '/log.txt', 'a') as f:
            f.write('\ntarget: {}\n'.format(pd))

        save_dir = os.path.join(opbase, pd)
        os.makedirs(save_dir, exist_ok=True)
        path_images = np.sort(glob(os.path.join(args.indir, pd, '*.tif')))

        # Each Criteria
        criteria_value = {}
        for c in criteria_list:
            criteria_value[c] = []

        with open(os.path.join(save_dir, 'criteria.csv'), 'w') as f:
            c = csv.writer(f)
            c.writerow(['time_point'] + criteria_list)

        tp = 0
        for pi in path_images:
            tp += 1
            criteria_current = [tp]
            img = io.imread(pi)
            center = np.array(img.shape * spatial_resolution) / 2

            # Number
            criteria_value['number'].append(len(np.unique(img)) - 1)
            criteria_current.append(len(np.unique(img)) - 1)

            # Volume
            volume_list = np.unique(img, return_counts=True)[1][1:] * voxel_convert_scale
            criteria_value['volume_sum'].append(np.sum(volume_list))
            criteria_current.append(np.sum(volume_list))
            criteria_value['volume_mean'].append(np.mean(volume_list))
            criteria_current.append(np.mean(volume_list))
            criteria_value['volume_sd'].append(np.std(volume_list))
            criteria_current.append(np.std(volume_list))

            # Surface Area, Aspect Ratio, Solidity, Centroid Coodinates
            ip_size = np.array(img.shape * spatial_resolution).astype(np.uint8)
            img_area = np.zeros(img.shape)
            centroid, aspect_ratio, solidity = [], [], []
            for n in range(1, len(np.unique(img))):
                img_bin = np.array(img == n) * 1
                img_ero = img_bin - morphology.erosion(img_bin, footprint=kernel)
                img_area += img_ero * n

                img_bin = np.array(transform.resize(img_bin * 255, ip_size, order=1, preserve_range=True) > 0) * 1
                p = measure.regionprops(img_bin)
                if len(p) == 0:
                    continue
                else:
                    p = p[0]
                try:
                    aspect_ratio.append(p.major_axis_length / p.minor_axis_length)
                except:
                    aspect_ratio.append(np.nan)
                try:
                    solidity.append(p.solidity)
                except:
                    solidity.append(np.nan)
                centroid.append(np.sqrt((center[2] - float(p.centroid[2])) ** 2 + (center[1] - float(p.centroid[1])) ** 2 + (center[0] - float(p.centroid[0])) ** 2))

            # Surface Area
            surface_list = np.unique(img_area, return_counts=True)[1][1:] * voxel_convert_scale
            criteria_value['surface_sum'].append(np.sum(surface_list))
            criteria_current.append(np.sum(surface_list))
            criteria_value['surface_mean'].append(np.mean(surface_list))
            criteria_current.append(np.mean(surface_list))
            criteria_value['surface_sd'].append(np.std(surface_list))
            criteria_current.append(np.std(surface_list))

            # Aspect Raio
            criteria_value['aspect_ratio_mean'].append(np.nanmean(aspect_ratio))
            criteria_current.append(np.nanmean(aspect_ratio))
            criteria_value['aspect_ratio_sd'].append(np.nanstd(aspect_ratio))
            criteria_current.append(np.nanstd(aspect_ratio))

            # Solidity
            criteria_value['solidity_mean'].append(np.nanmean(solidity))
            criteria_current.append(np.nanmean(solidity))
            criteria_value['solidity_sd'].append(np.nanstd(solidity))
            criteria_current.append(np.nanstd(solidity))

            # Centroid Coodinates
            criteria_value['centroid_mean'].append(np.mean(centroid))
            criteria_current.append(np.mean(centroid))
            criteria_value['centroid_sd'].append(np.std(centroid))
            criteria_current.append(np.std(centroid))


            with open(os.path.join(save_dir, 'criteria.csv'), 'a') as f:
                c = csv.writer(f)
                c.writerow(criteria_current)

            with open(opbase + '/log.txt', 'a') as f:
                f.write('tp {0:03d}: {1}\n'.format(tp, criteria_current))

        with open(os.path.join(save_dir, 'criteria.json'), 'w') as f:
            json.dump(criteria_value, f, indent=4)

if __name__ == '__main__':
    main()
