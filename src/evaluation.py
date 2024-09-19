# -*- coding: utf-8 -*-
# To Do : batch evaluation
import json
import numpy as np
import copy
import skimage.io as io
import os
import sys
from scipy.optimize import linear_sum_assignment
from skimage import morphology
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage import transform
from skimage import measure
from scipy import ndimage
from datetime import datetime
import pytz
from argparse import ArgumentParser
sys.path.append(os.getcwd())


def mucov(y, y_ans):
    # input -> pred(img), gt(img)
    sum_iou = 0
    label_list_y = np.unique(y)[1:]
    print('candidate label (pre): {}'.format(label_list_y))
    for i in label_list_y:
        y_mask = np.array((y == i) * 1).astype(np.int8)
        rp = measure.regionprops(y_mask)[0]
        bbox = rp.bbox
        y_ans_roi = y_ans[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        label_list = np.unique(y_ans_roi)[1:]
        best_iou, best_thr = 0, 0
        for j in label_list:
            y_ans_mask = np.array((y_ans == j) * 1).astype(np.int8)
            iou, thr = calc_iou(y_mask, y_ans_mask)
            if best_iou <= iou:
                best_iou = iou
                best_thr = np.max([thr, best_thr])
        print('c{0:03} best IoU in MUCov: {1}'.format(i, best_iou))
        if best_thr > 0.5:
            sum_iou += best_iou
        else:
            sum_iou += 0.0
    if len(label_list_y) == 0:
        return 0
    return sum_iou / len(label_list_y)

def seg(y, y_ans):
    # input -> pred(img), gt(img)
    sum_iou = 0
    label_list_y_ans = np.unique(y_ans)[1:]
    print('candidate label (gt): {}'.format(label_list_y_ans))
    for i in label_list_y_ans:
        y_ans_mask = np.array((y_ans == i) * 1).astype(np.int8)
        rp = measure.regionprops(y_ans_mask)[0]
        bbox = rp.bbox
        y_roi = y[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        label_list = np.unique(y_roi)[1:]
        best_iou, best_thr = 0, 0
        for j in label_list:
            y_mask = np.array((y == j) * 1).astype(np.int8)
            iou, thr = calc_iou(y_mask, y_ans_mask)
            if best_iou <= iou:
                best_iou = iou
                best_thr = np.max([thr, best_thr])
        print('c{0:03} best IoU in SEG: {1}'.format(i, best_iou))
        if best_thr > 0.5:
            sum_iou += best_iou
        else:
            sum_iou += 0.0
    return sum_iou / len(label_list_y_ans)

def calc_iou(pred, gt):
    # input -> pred(img), gt(img)
    pred, gt = pred.astype(np.int8), gt.astype(np.int8)
    countListPos = copy.deepcopy(pred + gt)
    countListNeg = copy.deepcopy(pred - gt)
    TP = len(np.where(countListPos.reshape(countListPos.size)==2)[0])
    FP = len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
    FN = len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])
    try:
        iou = TP / float(TP + FP + FN)
        thr = TP / float(TP + FN)
    except:
        iou = 0
        thr = 0
    return iou, thr

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
    if lab_img.min() == 1:
        remove_voxel = mask_size[lab_img-1]
    else:
        remove_voxel = mask_size[lab_img]
    lab_img[remove_voxel] = 0

    labels = np.unique(lab_img)
    lab_img = np.searchsorted(labels, lab_img)
    return lab_img

def eval_metrics(TP, FP, FN, TN=None):
    evals = {}
    try:
        evals['accuracy'] = (TP + TN) / float(TP + TN + FP + FN)
    except:
        evals['accuracy'] = 0.0
    try:
        evals['recall'] = TP / float(TP + FN)
    except:
        evals['recall'] = 0.0
    try:
        evals['precision'] = TP / float(TP + FP)
    except:
        evals['precision'] = 0.0
    try:
        evals['specificity'] = TN / float(TN + FP)
    except:
        evals['specificity'] = 0.0
    try:
        evals['F-measure'] = 2 * evals['recall'] * evals['precision'] / (evals['recall'] + evals['precision'])
    except:
        evals['F-measure'] = 0.0
    try:
        evals['IoU'] = TP / float(TP + FP + FN)
    except:
        evals['IoU'] = 0.0
    return evals


class NDNEvaluator():
    def __init__(self, connectivity=1, radius=10, delv=3, criteria=['recall', 'precision', 'F-measure', 'IoU']):
        super().__init__()
        self.connectivity=connectivity
        self.radius = radius
        self.criteria = criteria
        self.delv = delv

    def calc_TP(self, pre_img, gt_img):
        PR = self._extract_centroids(pre_img)
        GT = self._extract_centroids(gt_img)
        #input -> PR, GT: coordinates of centroid
        numPR, numGT = len(PR), len(GT)
        if numPR == 0:
            return 0, numPR, numGT

        cost = []
        for pr in PR:
            cost.append([int(np.sum((gt - pr)**2) < self.radius**2) for gt in GT])
            
        rows, cols = linear_sum_assignment(cost, maximize=True)

        TP = 0
        for row, col in zip(rows, cols):
            if cost[row][col] == 1:
                TP += 1

        FP = numPR - TP
        FN = numGT - TP

        return TP, FP, FN

    def evaluation(self, det, gt):
        det = labeling(det, connectivity=self.connectivity, delv=self.delv)
        gt = labeling(gt, connectivity=self.connectivity, delv=0)
        TP, FP, FN = self.calc_TP(det, gt)
        evals = eval_metrics(TP=TP, FP=FP, FN=FN)
        results = {}
        for cri in self.criteria:
            results[cri] = evals[cri]
        return results

    def _extract_centroids(self, det):
        det_props = regionprops(det)
        det_centroid=np.array([prop.centroid for prop in det_props])
        return det_centroid

    def _search_list(self, node, used, idx):
        if len(node) == idx:
            return 0
        else:
            tmp = []
            for i in range(len(node[idx])):
                if used[node[idx][i]] == 0:
                    used[node[idx][i]] += 1
                    tmp.append(self._search_list(node, used, idx+1) + 1)
                    used[node[idx][i]] -= 1
                else:
                    tmp.append(self._search_list(node, used, idx+1))
            return np.max(tmp)


class NSNEvaluator():
    def __init__(self, criteria=['IoU', 'MUCov', 'SEG'], connectivity=0):
        super().__init__()
        self.connectivity=connectivity
        self.criteria = criteria

    def calc_TP(self, pre_img, gt_img): 
        pred, gt = pre_img.astype(np.int8), gt_img.astype(np.int8)
        countListPos = copy.deepcopy(pred + gt)
        countListNeg = copy.deepcopy(pred - gt)
        TP = len(np.where(countListPos.reshape(countListPos.size)==2)[0])
        TN = len(np.where(countListPos.reshape(countListPos.size)==0)[0])
        FP = len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
        FN = len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])
        return TP, TN, FP, FN
    
    def evaluation(self, pred, gt):
        pre_lab = labeling(pred, connectivity=self.connectivity)
        gt_lab = labeling(gt, connectivity=self.connectivity)
        pre_bin = np.array((pre_lab > 0) * 1).astype(np.int8)
        gt_bin = np.array((gt_lab > 0) * 1).astype(np.int8)
        TP, TN, FP, FN  = self.calc_TP(pre_bin, gt_bin)
        evals = eval_metrics(TP=TP, FP=FP, FN=FN, TN=TN)
        if 'MUCov' in self.criteria:
            evals['MUCov'] = mucov(pre_lab, gt_lab)
        if 'SEG' in self.criteria:
            evals['SEG'] = seg(pre_lab, gt_lab)
        results = {}
        for cri in self.criteria:
            results[cri] = evals[cri]
        return results

class ClassificationEvaluator():
    def __init__(self, criteria=['accuracy']):
        super().__init__()
    
    def evaluation(self, lab, gt):
        score = (lab == gt) * 1
        return {'accuracy': np.mean(score)}
    

if __name__ == '__main__':
    ap = ArgumentParser(description='python evaluation.py')
    ap.add_argument('--indir', '-i', nargs='?', default='results/test_of_instance_segmentation_Sep21Tue_2021_035232/SegmentationImages', help='Specify input files')
    #ap.add_argument('--gtdir', '-g', nargs='?', default='../brightfieldsegmentation/images/nsn', help='Specify ground truth files')
    ap.add_argument('--gtdir', '-g', nargs='?', default='/data/kanazawa/images/qcanet', help='Specify ground truth files')
    ap.add_argument('--outdir', '-o', nargs='?', default='evaluation_of_segmentation', help='Specify output files directory for create figures')
    ap.add_argument('--labeling', type=int, default=1, help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--task', type=str, default='segmentation', help='Specify task (ndn or nsn or classification)')
    args = ap.parse_args()
    argvs = sys.argv
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    opbase = args.outdir + '_' + current_datetime
    os.makedirs(opbase, exist_ok=True)

    with open(opbase + '/result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    if args.task == 'detection':
        evaluator = NDNEvaluator(connectivity=args.labeling)
        evals_all = {'F-measure':[], 'recall':[], 'precision':[], 'IoU':[]}
    elif args.task == 'segmentation':
        evaluator = NSNEvaluator(connectivity=args.labeling)
        evals_all = {'IoU':[], 'MUCov':[], 'SEG':[]}
    elif args.task == 'classification':
        evaluator = ClassificationEvaluator()
        evals_all = {'accuracy':[]}
    else:
        print('Warning: select segmentation or detection or classification')
        sys.exit()

    emb_list = np.sort(os.listdir(args.indir))
    for emb in emb_list:
        seg_dir = args.indir + '/' + emb
        gt_dir = args.gtdir + '/' + emb.replace('_', '/')
        #gt_dir = args.gtdir + '/' + emb

        in_list = np.sort(os.listdir(seg_dir))
        gt_list = np.sort(os.listdir(gt_dir))
        #assert len(in_list) == len(gt_list)

        with open(opbase + '/result.txt', 'a') as f:
            for t in in_list:
                tp = t[:t.rfind('.')]
                print('#################')
                print('Input file name: {}'.format(os.path.join(args.indir, t)))
                print('GT file name   : {}'.format(os.path.join(gt_dir, t)))
                print('#################')
                pre = io.imread(os.path.join(seg_dir, t)).astype(np.uint16)
                gt = io.imread(os.path.join(gt_dir, t)).astype(np.uint16)
                evals = evaluator.evaluation(pre, gt)

                print('#################')
                print('Embryo:{}'.format(emb))
                print('Timepoint:{}'.format(tp))
                f.write('Embryo:{}\n'.format(emb))
                f.write('Timepoint:{}\n'.format(tp))

                for cri, val in evals.items():
                    evals_all[cri].append(val)
                    print('tp{}: {}={}'.format(tp, cri, val))
                    f.write('tp{}: {}={}\n'.format(tp, cri, val))
                print('#################')
                print('')

    results = {}
    for cri, val in evals_all.items():
        results[cri] = np.mean(val)
        results[cri + '_std'] = np.std(val)
    with open(os.path.join(opbase, 'result.json'), 'w') as f:
        json.dump(results, f)
