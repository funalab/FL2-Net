# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import copy
import re
import time
import argparse
import skimage.io as io
from os import path as pt
from distutils.util import strtobool
import torch
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import find_objects


def get_optimizer(cfg, model):
    ''' Optimizer '''
    # Initialize an optimizer
    if cfg.RUNTIME.OPTIMIZER.NAME == 'SGD':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            momentum=cfg.RUNTIME.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
            )
    elif cfg.RUNTIME.OPTIMIZER.NAME == 'Adadelta':
        optimizer = optim.Adadelta(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            rho=cfg.RUNTIME.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
            )
    elif cfg.RUNTIME.OPTIMIZER.NAME == 'Adagrad':
        optimizer = optim.Adagrad(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
            )
    elif cfg.RUNTIME.OPTIMIZER.NAME == 'Adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            amsgrad=False,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
            )
    elif cfg.RUNTIME.OPTIMIZER.NAME == 'AdamW':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
            )
    elif cfg.RUNTIME.OPTIMIZER.NAME == 'AMSGrad':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=cfg.RUNTIME.OPTIMIZER.INIT_LR,
            amsgrad=True,
            weight_decay=cfg.RUNTIME.OPTIMIZER.WEIGHT_DECAY
        )

    return optimizer


def print_args(dataset_args, model_args, runtime_args):
    """ Export config file
    Args:
        dataset_args    : Argument Namespace object for loading dataset
        model_args        : Argument Namespace object for model parameters
        runtime_args    : Argument Namespace object for runtime parameters
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    print('============================')
    print('[Dataset]')
    for k, v in dataset_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Model]')
    for k, v in model_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Runtime]')
    for k, v in runtime_dict.items():
        print('%s = %s' % (k, v))
    print('============================\n')


def loadImages(path):
    imagePathes = map(lambda a:os.path.join(path,a),os.listdir(path))
    try:
        imagePathes.pop(imagePathes.index(path + '/.DS_Store'))
    except:
        pass
    imagePathes = np.sort(imagePathes) # list on Linux
    images = np.array(map(lambda x: io.imread(x), imagePathes))
    return images


def oneSideExtensionImage(images, patchsize):
    lz, ly, lx = images.shape
    if lx % patchsize != 0:
        sx = lx + patchsize - 1
    else:
        sx = lx
    if ly % patchsize != 0:
        sy = ly + patchsize - 1
    else:
        sy = ly
    if lz % patchsize != 0:
        sz = lz + patchsize - 1
    else:
        sz = lz
    exbox = np.zeros((sz, sy, sx))
    exbox += images.min()
    exbox[0:lz, 0:ly, 0:lx] = images
    return copy.deepcopy(exbox)


def patch_crop(x_data, y_data, idx, n, patchsize):
    x_patch = copy.deepcopy( np.array( x_data[ idx[n][2]:idx[n][2]+patchsize, idx[n][1]:idx[n][1]+patchsize, idx[n][0]:idx[n][0]+patchsize ] ).reshape(1, patchsize, patchsize, patchsize).astype(np.float32) )   # np.shape(idx_O[n][0]) [0] : x座標, n : 何番目の座標か
    y_patch = copy.deepcopy( np.array(y_data[ idx[n][2] ][ idx[n][1] ][ idx[n][0] ]).reshape(1).astype(np.int32) )
    return x_patch, y_patch


def crossSplit(objData, bgData, objLabel, bgLabel, k_cross, n):
    objx, bgx, objy, bgy = [], [], [], []
    N = len(objData)
    for i in range(k_cross):
        objx.append(objData[i*N/k_cross:(i+1)*N/k_cross])
        objy.append(objLabel[i*N/k_cross:(i+1)*N/k_cross])
        bgx.append(bgData[i*N/k_cross:(i+1)*N/k_cross])
        bgy.append(bgLabel[i*N/k_cross:(i+1)*N/k_cross])
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(k_cross):
        if i == n:
            x_test.append(copy.deepcopy(objx[i]))
            x_test.append(copy.deepcopy(bgx[i]))
            y_test.append(copy.deepcopy(objy[i]))
            y_test.append(copy.deepcopy(bgy[i]))
        else:
            x_train.append(copy.deepcopy(objx[i]))
            x_train.append(copy.deepcopy(bgx[i]))
            y_train.append(copy.deepcopy(objy[i]))
            y_train.append(copy.deepcopy(bgy[i]))
    x_train = np.array(x_train).reshape(2*N*(k_cross-1)/k_cross, args.batchsize, patchsize, patchsize, patchsize)
    y_train = np.array(y_train).reshape(2*N*(k_cross-1)/k_cross, args.batchsize)
    x_test = np.array(x_test).reshape(2*N/k_cross, args.batchsize, patchsize, patchsize, patchsize)
    y_test = np.array(y_test).reshape(2*N/k_cross, args.batchsize)
    return copy.deepcopy(x_train), copy.deepcopy(x_test), copy.deepcopy(y_train), copy.deepcopy(y_test)


# Rotation & Flip for Data Augmentation (fix z-axis)
def dataAugmentation(image, rot=True, flip=True):
    lz, ly, lx = image.shape
    if rot and flip:
        flip = np.zeros((lz, ly, lx))
        rot90 = np.zeros((lz, lx, ly))
        rot90_f = np.zeros((lz, lx, ly))
        rot180 = np.zeros((lz, ly, lx))
        rot180_f = np.zeros((lz, ly, lx))
        rot270 = np.zeros((lz, lx, ly))
        rot270_f = np.zeros((lz, lx, ly))
        for z in range(lz):
            flip[z] = np.flip(image[z], 1)
            rot90[z] = np.rot90(image[z])
            rot90_f[z] = np.rot90(flip[z])
            rot180[z] = np.rot90(rot90[z])
            rot180_f[z] = np.rot90(rot90_f[z])
            rot270[z] = np.rot90(rot180[z])
            rot270_f[z] = np.rot90(rot180_f[z])
        aug_images = [flip, rot90, rot90_f, rot180, rot180_f, rot270, rot270_f]
    elif flip:
        flip_v = np.zeros((lz, ly, lx))
        flip_h = np.zeros((lz, ly, lx))
        flip_vh = np.zeros((lz, ly, lx))
        for z in range(lz):
            flip_v[z] = np.flip(image[z], 0)
            flip_h[z] = np.flip(image[z], 1)
            flip_vh[z] = np.flip(flip_h[z], 0)
        aug_images = [flip_v, flip_h, flip_vh]
    elif rot:
        rot90 = np.zeros((lz, lx, ly))
        rot180 = np.zeros((lz, ly, lx))
        rot270 = np.zeros((lz, lx, ly))
        for z in range(lz):
            rot90[z] = np.rot90(image[z])
            rot180[z] = np.rot90(rot90[z])
            rot270[z] = np.rot90(rot180[z])
        aug_images = [rot90, rot180, rot270]
    else:
        print('No Augmentation!')
        aug_images = []
    return aug_images


# Create Opbase for Output Directory
def createOpbase(opbase):
    if (opbase[len(opbase) - 1] == '/'):
        opbase = opbase[:len(opbase) - 1]
    if not (opbase[0] == '/'):
        if (opbase.find('./') == -1):
            opbase = './' + opbase
    t = time.ctime().split(' ')
    if t.count('') == 1:
        t.pop(t.index(''))
    opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
    if not (pt.exists(opbase)):
        os.mkdir(opbase)
        print('Output Directory not exist! Create...')
    print('Output Directory: {}'.format(opbase))
    return opbase


# Oneside Mirroring Padding in Image-wise Processing
def mirror_extension_image(image, ndim=3, length=10, tdim=False, mode='reflect'):
    pad_none = np.zeros((image.ndim - ndim, 2), np.int16)
    pads = np.array(np.array([[length, length]] * ndim), np.int16)
    pad_len = np.concatenate([pad_none, pads])
    exbox = np.pad(image, pad_width=pad_len, mode=mode)
    return exbox

def splitImage(image, stride):
    lz, ly, lx = np.shape(image)
    num_split = int(((lx - stride) / stride) * ((ly - stride) / stride) * ((lz - stride) / stride))
    s_image = np.zeros((num_split, self.patchsize, self.patchsize, self.patchsize))
    num = 0
    for z in range(0, lz-stride, stride):
        for y in range(0, ly-stride, stride):
            for x in range(0, lx-stride, stride):
                s_image[num] = image[z:z+self.patchsize, y:y+self.patchsize, x:x+self.patchsize]
                num += 1
    return copy.deepcopy(s_image)

def load_csv(gtdir, filename, lab_num=3):
    tp = int(os.path.splitext(os.path.basename(filename))[0])
    emb = int(re.sub(r'\D', "", os.path.basename(os.path.dirname(filename))))
    field = os.path.dirname(os.path.dirname(filename))
    gt_list = np.loadtxt(os.path.join(gtdir, field) + '.csv', skiprows=1, delimiter=',', usecols=(0, 1, 2)).astype(np.int)
    cell_stages = gt_list[np.where(gt_list[:, 0] == emb)][0, 1:]
    for i in range(lab_num):
        if i == (lab_num - 1):
            label = i
            break
        elif tp < cell_stages[i]:
            label = i
            break
    return label

def labeling(mask, marker=None, connectivity=1, delv=0):
    if connectivity == 0:
        return mask
    elif marker != None:
        distance = ndimage.distance_transform_edt(mask)
        lab_marker = morphology.label(marker, connectivity=connectivity)
        lab_img = watershed(-distance, lab_marker, mask=mask)
    else:
        lab_img = morphology.label(mask, connectivity=connectivity)
    # delete small object
    mask_size = np.unique(lab_img, return_counts=True)[1] < (delv + 1)
    remove_voxel = mask_size[lab_img]
    lab_img[remove_voxel] = 0

    labels = np.unique(lab_img)
    lab_img = np.searchsorted(labels, lab_img)
    return lab_img


def to_one_hot(instance, del_bg=True):
    """convert label image to one-hot vector

    Params:
        instance: labeled tensor of dim [D, H, W]
        del_bg: whether to delete background channel (instance == 0) or not
    Return:
        x_one_hot: one-hot tensor image of dim [N, D, H, W]
            (where N is the number of ground-truth instnaces in the target)
    """
    d, h, w = instance.shape
    ids = torch.unique(instance)
    n_instance = len(ids)

    x_expand = instance.expand(n_instance, d, h, w)
    ids_expand = ids[:, None, None, None].expand(n_instance, d, h, w)
    x_one_hot = x_expand == ids_expand
    if del_bg:
        x_one_hot = x_one_hot[1:]
    return x_one_hot.bool()

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, D, H, W) tenor to (N, DxHxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def edt_prob(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape,np.float32)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        # 1. grow object slice by 1 for all interior object bounding boxes
        # 2. perform (correct) EDT for object with label id i
        # 3. extract EDT for object of original slice and normalize
        # 4. store edt for object only for pixels of given label id i
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask = grown_mask[shrink_slice]
        edt = distance_transform_edt(grown_mask, sampling=anisotropy)[shrink_slice][mask]
        prob[sl][mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
    return prob

