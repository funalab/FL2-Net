# -*- coding: utf-8 -*-

import os
import sys
import re
import random
import numpy as np
from skimage import io
from skimage import io
from skimage import transform as tr
from skimage.segmentation import relabel_sequential
from skimage import morphology

from src.utils.utils import mirror_extension_image
import torch
from torch.utils.data import Dataset

sys.setrecursionlimit(10000)

def read_img(path, arr_type='npz'):
    """ read image array from path
    Args:
        path (str)          : path to directory which images are stored.
        arr_type (str)      : type of reading file {'npz','jpg','png','tif'}
    Returns:
        image (np.ndarray)  : image array
    """
    if arr_type == 'npz':
        image = np.load(path)['arr_0']
    elif arr_type in ('png', 'jpg'):
        image = imread(path, mode='L')
    elif arr_type == 'tif':
        image = io.imread(path)
    else:
        raise ValueError('invalid --input_type : {}'.format(arr_type))

    return image.astype(np.int32)


def crop_pair_3d(
        image1,
        image2,
        crop_size=(128, 128, 128),
        nonzero_image1_thr=0.000001,
        nonzero_image2_thr=0.000001,
        augmentation=True
    ):
    """ 3d {image, label} patches are cropped from array.
    Args:
        image1 (np.ndarray)                  : Input 3d image array from 1st domain
        image2 (np.ndarray)                  : Input 3d label array from 2nd domain
        crop_size ((int, int, int))         : Crop image patch from array randomly
        nonzero_image1_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
        nonzero_image2_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
    Returns:
        cropped_image1 (np.ndarray)  : cropped 3d image array
        cropped_image2 (np.ndarray)  : cropped 3d label array
    """
    z_len, y_len, x_len = image1.shape[-3:]
    assert x_len >= crop_size[0]
    assert y_len >= crop_size[1]
    assert z_len >= crop_size[2]

    while 1:
        # get cropping position (image)
        top = random.randint(0, x_len-crop_size[0]-1) if x_len > crop_size[0] else 0
        left = random.randint(0, y_len-crop_size[1]-1) if y_len > crop_size[1] else 0
        front = random.randint(0, z_len-crop_size[2]-1) if z_len > crop_size[2] else 0
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        rear = front + crop_size[2]

        # crop image
        cropped_image1 = image1[..., front:rear, left:right, top:bottom]
        cropped_image2 = image2[..., front:rear, left:right, top:bottom]
        # get nonzero ratio
        nonzero_image1_ratio = np.nonzero(cropped_image1)[0].size / float(cropped_image1.size)
        nonzero_image2_ratio = np.nonzero(cropped_image2)[0].size / float(cropped_image2.size)
        
        # rotate {image_A, image_B}
        if augmentation:
            aug_flag = random.randint(0, 3)
            cropped_image1 = np.rot90(cropped_image1, k=aug_flag, axes=(-2, -1))
            cropped_image2 = np.rot90(cropped_image2, k=aug_flag, axes=(-2, -1))
        
        # break loop
        if (nonzero_image1_ratio >= nonzero_image1_thr) \
                and (nonzero_image2_ratio >= nonzero_image2_thr):
            return cropped_image1.copy(), cropped_image2.copy(), (top, left, front), aug_flag

def crop_3d(
        image,
        crop_size=(128, 128, 128),
        augmentation = True,
        position=None,
        aug_flag=None
    ):
    """ 3d image patche is cropped from array.
    Args:
        image (np.ndarray)                  : Input 3d image array
        top, left, front                    : Position of crop image 
        crop_size ((int, int, int))         : Crop image patch from array randomly
    Returns:
        cropped_image (np.ndarray)  : cropped 3d image array
    """
    z_len, y_len, x_len = image.shape[-3:]
    assert x_len >= crop_size[0]
    assert y_len >= crop_size[1]
    assert z_len >= crop_size[2]

    # get cropping position (image)
    if position is not None:
        top, left, front = position
    else:  
        top = random.randint(0, x_len-crop_size[0]-1) if x_len > crop_size[0] else 0
        left = random.randint(0, y_len-crop_size[1]-1) if y_len > crop_size[1] else 0
        front = random.randint(0, z_len-crop_size[2]-1) if z_len > crop_size[2] else 0
        
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    rear = front + crop_size[2]

    # crop image
    cropped_image = image[..., front:rear, left:right, top:bottom]

    # rotate image
    if augmentation:
        if aug_flag is None:
            aug_flag = random.randint(0, 3)
        cropped_image = np.rot90(cropped_image, k=aug_flag, axes=(-2, -1))
    
    return cropped_image.copy(), (top, left, front), aug_flag


class PreprocessedDataset(Dataset):

    def __init__(self, cfg, train=False):
        self.raw_path = cfg.DATASETS.DIR_NAME.RAW
        self.multi_task = False
        self.task = cfg.TASK
        self.center = False
        self.gt_path = cfg.DATASETS.DIR_NAME.get('INSTANCE', None)
        if cfg.MODEL.META_ARCHITECTURE == "EmbedSeg" and train:
            self.center = True
            self.center_path = cfg.DATASETS.DIR_NAME.CENTER
        self.split_list = cfg.DATASETS.SPLIT_LIST
        self.arr_type = cfg.DATASETS.ARRAY_TYPE
        self.augmentation = cfg.DATASETS.PREPROCESS.AUGMENTATION
        self.scaling = cfg.DATASETS.PREPROCESS.SCALING
        self.resolution = cfg.DATASETS.RESOLUTION
        self.crop_size = cfg.DATASETS.INPUT_SIZE
        self.ndim = cfg.DATASETS.DIMMENSION
        self.rnn = cfg.MODEL.BACKBONE.RECURRENT
        self.train = train
        self.base_time = cfg.DATASETS.get('BASETIME', 1)
        if self.rnn:
            self.seq_len = cfg.MODEL.BACKBONE.RNN.LENGTH
            self.bidirectional = cfg.MODEL.BACKBONE.RNN.BIDIRECTIONAL
        else:
            self.seq_len = 1
            self.bidirectional = False
        with open(cfg.DATASETS.SPLIT_LIST, 'r') as f:
            self.img_path = f.read().split()

    def __len__(self):
        return len(self.img_path)
    
    def _normalize(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        return image

    def _resize(self, image, order=0):
        img_shape = image.shape[-3:]
        ip_size = (int(img_shape[0] * self.resolution[2]), int(img_shape[1] * self.resolution[1]), int(img_shape[2] * self.resolution[0]))
        input_shape = image.shape[:image.ndim - 3] + ip_size
        image = tr.resize(image, input_shape, order=order, preserve_range=True)
        return image
    
    def _padding(self, image, mode='reflect'):
        ip_size = image.shape[-3:]
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            image = mirror_extension_image(image, ndim=3, length=pad_size, mode=mode)
        return image

    def _padding_min(self, image, mode='reflect', img_pads=None):
        img_size = image.shape[-3:]
        pads = [(0, 0) for i in range(image.ndim - self.ndim)]
        if img_pads is None:
            img_pads = []
            for i, c in zip(img_size, reversed(self.crop_size)):
                pad_a = random.randint(0, c-i) if i < c else 0
                pad_b = (c - i) - pad_a if i < c else 0
                img_pads.append((pad_a, pad_b))

        pads = pads + img_pads
        image = np.pad(image, pad_width=pads, mode=mode)
        return image, img_pads

    def __getitem__(self, i):
        '''
        [output]
        segmentation/detection: raw_img + gt_img
        [shape of gt_img] -> (C, D, H, W)
            bidirectional(or attention): seq_gt[seq_len//2]
            other: seq_gt[-1]
        [shape of raw_img] -> (L, C, D, H, W)
        '''
        print(self.img_path[i])
        inputs = {}

        x = []
        first_tp = int(os.path.splitext(os.path.basename(self.img_path[i]))[0]) - self.seq_len + 1
        emb_name = os.path.dirname(self.img_path[i])
        
        # get raw image
        for tp in range(self.seq_len):
            if first_tp + tp < self.base_time:
                img_path = os.path.join(emb_name, '{0:03d}.tif'.format(self.base_time))
            else:
                img_path = os.path.join(emb_name, '{0:03d}.tif'.format(first_tp + tp))
            x.append(self._normalize(
                read_img(os.path.join(self.raw_path, img_path), self.arr_type)
            ))
        x = np.stack(x)

        # store image properties
        inputs['filename'] = self.img_path[i]
        inputs['image_size'] = np.array(x.shape[-3:])
        print(self.img_path[i])

        x = self._resize(self._normalize(x), order=1)
        x = np.expand_dims(x, axis=1).astype(np.float32)

        if not self.rnn:
            x = x.squeeze(0)

        # get ground truth (image and/or class)
        if self.bidirectional:
            if first_tp + self.seq_len // 2 < self.base_time:
                img_path_gt = os.path.join(emb_name, '{0:03}.tif'.format(self.base_time))
            else:
                img_path_gt = os.path.join(emb_name, '{0:03d}.tif'.format(first_tp + self.seq_len // 2))
        else:
            img_path_gt = os.path.join(emb_name, '{0:03d}.tif'.format(first_tp + self.seq_len - 1))

        if self.gt_path == None:
            img_path_gt = None
        else:
            gt_img = read_img(os.path.join(self.gt_path, img_path_gt), self.arr_type).astype(np.int32)

        if self.train:
            # padding
            x = self._padding(x)
            gt_img = self._padding(self._resize(gt_img, order=0))

            # crop image
            inputs['image'], instance, position, aug_flag = crop_pair_3d(x, gt_img, crop_size=self.crop_size)
            instance = morphology.label(instance, connectivity=3)
            inputs['instances'] = np.zeros((instance.shape[-3], instance.shape[-2], instance.shape[-1]), dtype=np.int16)
            mask = instance > 0
            inputs['instances'][mask] = relabel_sequential(instance[mask])[0]

            if self.center:
                center_img = read_img(os.path.join(self.center_path, img_path_gt), self.arr_type).astype(np.int32)
                center_img = self._padding(self._resize(center_img, order=0))
                inputs['centers'], _, _ = crop_3d(center_img, crop_size=self.crop_size, position=position, aug_flag=aug_flag)

        elif self.gt_path != None:
            # multi_batch inference is not supported
            inputs['image'] = x
            inputs['instances'] = np.zeros((gt_img.shape[-3], gt_img.shape[-2], gt_img.shape[-1]), dtype=np.int16)
            mask = gt_img > 0
            inputs['instances'][mask] = relabel_sequential(gt_img[mask])[0]

        else:
            inputs['image'] = x

        return inputs
