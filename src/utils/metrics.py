"""
Modified from StarDist `matching.py`
https://github.com/stardist/stardist/blob/master/stardist/matching.py
"""

from collections import namedtuple

import numpy as np
#from numba import jit
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys
from datetime import datetime
import pytz
import skimage.io as io
import json

matching_criteria = dict()


def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label='labels' if name is None else name,
        integers=('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _raise(e):
    raise e


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def relabel_sequential(label_field, offset=1):
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    m = label_field.max()
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(int(m))
        label_field = label_field.astype(new_type)
        m = m.astype(new_type)  # Ensures m is an integer
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    required_type = np.min_scalar_type(offset + len(labels0))
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        label_field = label_field.astype(required_type)
    new_labels0 = np.arange(offset, offset + len(labels0))
    if np.all(labels0 == new_labels0):
        return label_field, labels, labels
    forward_map = np.zeros(int(m + 1), dtype=label_field.dtype)
    forward_map[labels0] = new_labels0
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = np.zeros(offset - 1 + len(labels), dtype=label_field.dtype)
    inverse_map[(offset - 1):] = labels
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x, 'x', True)
        _check_label_array(y, 'y', True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)


#@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def intersection_over_union(overlap):
    _check_label_array(overlap, 'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return overlap / (n_pixels_pred + n_pixels_true - overlap)


matching_criteria['iou'] = intersection_over_union


def matching_dataset(y_true, y_pred, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):
    len(y_true) == len(y_pred) or _raise(ValueError("y_true and y_pred must have the same length."))
    return matching_dataset_lazy(
        tuple(zip(y_true, y_pred)), thresh=thresh, criterion=criterion, by_image=by_image, show_progress=show_progress,
        parallel=parallel,
    )


def matching_dataset_lazy(y_gen, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):
    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true',
                         'n_pred', 'mean_true_score'))

    single_thresh = False
    if np.isscalar(thresh):
        single_thresh = True
        thresh = (thresh,)

    tqdm_kwargs = {}
    tqdm_kwargs['disable'] = not bool(show_progress)
    if int(show_progress) > 1:
        tqdm_kwargs['total'] = int(show_progress)

    # compute matching stats for every pair of label images
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        fn = lambda pair: matching(*pair, thresh=thresh, criterion=criterion, report_matches=False)
        with ThreadPoolExecutor() as pool:
            stats_all = tuple(pool.map(fn, tqdm(y_gen, **tqdm_kwargs)))
    else:
        stats_all = tuple(
            matching(y_t, y_p, thresh=thresh, criterion=criterion, report_matches=False)
            for y_t, y_p in tqdm(y_gen, **tqdm_kwargs)
        )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i, s in enumerate(stats):
            acc = accumulate[i]
            for k, v in s._asdict().items():
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_true_score"
                    acc[k] = acc.setdefault(k, 0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k, 0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr, acc in zip(thresh, accumulate):
        set(acc.keys()) == expected_keys or _raise(ValueError("unexpected keys"))
        acc['criterion'] = criterion
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)
        if bool(by_image):
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
                acc[k] /= n_images
        else:
            tp, fp, fn = acc['tp'], acc['fp'], acc['fn']
            acc.update(
                precision=precision(tp, fp, fn),
                recall=recall(tp, fp, fn),
                accuracy=accuracy(tp, fp, fn),
                f1=f1(tp, fp, fn),
                mean_true_score=acc['mean_true_score'] / acc['n_true'] if acc['n_true'] > 0 else 0.0,
            )

    accumulate = tuple(namedtuple('DatasetMatching', acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate


def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """
    if report_matches=True, return (matched_pairs,matched_scores) are independent of 'thresh'
    """

    _check_label_array(y_true, 'y_true')
    _check_label_array(y_pred, 'y_pred')
    y_true.shape == y_pred.shape or _raise(ValueError(
        "y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true,
                                                                                           y_pred=y_pred)))
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))
    if thresh is None: thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float, thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind, pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        stats_dict = dict(
            criterion=criterion,
            thresh=thr,
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision(tp, fp, fn),
            recall=recall(tp, fp, fn),
            accuracy=accuracy(tp, fp, fn),
            f1=f1(tp, fp, fn),
            n_true=n_true,
            n_pred=n_pred,
            mean_true_score=np.sum(scores[true_ind, pred_ind][match_ok]) / n_true if not_trivial else 0.0,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update(
                    # int() to be json serializable
                    matched_pairs=tuple(
                        (int(map_rev_true[i]), int(map_rev_pred[j])) for i, j in zip(1 + true_ind, 1 + pred_ind)),
                    matched_scores=tuple(scores[true_ind, pred_ind]),
                    matched_tps=tuple(map(int, np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update(
                    matched_pairs=(),
                    matched_scores=(),
                    matched_tps=(),
                )
        return namedtuple('Matching', stats_dict.keys())(*stats_dict.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single, thresh))


def obtain_APdsb_one_hot(gt_image, prediction_image, ap_val):
    """
        Obtain Average Precision (AP_dsb) or Accuracy for GT, one-hot label masks

        Parameters
        -------

        gt_image: numpy array
            label mask (IYX)
        prediction_image: numpy array
            label mask (YX)
        ap_val: float
            IoU Threshold between 0 and 1.0

        Returns
        -------
        score: float
            Average Precision (AP_dsb) or Accuracy

        """

    gt_ids = np.arange(gt_image.shape[0])
    prediction_ids = np.unique(prediction_image)[1:]  # ignore background
    iou_table = np.zeros((len(prediction_ids), len(gt_ids)))

    for j in range(iou_table.shape[0]):
        for k in range(iou_table.shape[1]):
            intersection = ((gt_image[k] > 0) & (prediction_image == prediction_ids[j]))
            union = ((gt_image[k] > 0) | (prediction_image == prediction_ids[j]))
            iou_table[j, k] = np.sum(intersection) / np.sum(union)

    iou_table_binary = iou_table >= ap_val
    FP = np.sum(np.sum(iou_table_binary, axis=1) == 0)
    FN = np.sum(np.sum(iou_table_binary, axis=0) == 0)
    TP = iou_table_binary.shape[1] - FN
    score = TP / (TP + FP + FN)
    return score


if __name__ == '__main__':
    ap = ArgumentParser(description='python metrics.py')
    ap.add_argument('--indir', '-i', nargs='?', default='results/test_of_instance_segmentation_Sep21Tue_2021_035232/SegmentationImages', help='Specify input files')
    ap.add_argument('--gtdir', '-g', nargs='?', default='/data/kanazawa/images/qcanet', help='Specify ground truth files')
    ap.add_argument('--outdir', '-o', nargs='?', default='results/evaluation_of_segmentation', help='Specify output files directory for create figures')

    args = ap.parse_args()
    argvs = sys.argv
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    opbase = args.outdir + '_' + current_datetime
    os.makedirs(opbase, exist_ok=True)

    #threshold = np.linspace(50, 90, 9)
    threshold = np.linspace(10, 90, 9).astype(np.int16)
    AP_dsb = {str(thr):[] for thr in threshold}
    
    field = ('criterion',
             'thresh',
             'fp',
             'tp',
             'fn',
             'precision',
             'recall',
             'accuracy',
             'f1',
             'n_true',
             'n_pred',
             'mean_true_score',
             'by_image')

    criteria = [
        'precision',
        'recall',
        'accuracy',
        'f1',
        'mean_true_score',
    ]

    results_all = {cri+'_'+str(thr):{} for cri in criteria for thr in threshold}
    
    with open(opbase + '/result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    emb_list = np.sort(os.listdir(args.indir))
    for emb in emb_list:
        seg_dir = args.indir + '/' + emb
        gt_dir = args.gtdir + '/' + emb.replace('_', '/')

        in_list = np.sort(os.listdir(seg_dir))
        gt_list = np.sort(os.listdir(gt_dir))

        with open(opbase + '/result.txt', 'a') as f:
            for t in in_list:
                tp = t[:t.rfind('.')]
                print('#################')
                print('Input file name: {}'.format(os.path.join(seg_dir, t)))
                print('GT file name   : {}'.format(os.path.join(gt_dir, t)))
                print('#################')
                emb_name = emb + '/' + tp
                pre = io.imread(os.path.join(seg_dir, t)).astype(np.uint16)
                gt = io.imread(os.path.join(gt_dir, t)).astype(np.uint16)
                results = matching_dataset([gt], [pre], thresh=threshold/100)

                print('#################')
                print('Embryo:{}'.format(emb))
                print('Timepoint:{}'.format(tp))
                f.write('Embryo:{}\n'.format(emb))
                f.write('Timepoint:{}\n'.format(tp))

                for res, thr in zip(results, threshold):
                    for cri in criteria:
                        res_dict = res._asdict()
                        print('tp{}: {}({})={}'.format(tp, cri, str(thr), res_dict[cri]))
                        f.write('tp{}: {}({})={}\n'.format(tp, cri, str(thr), res_dict[cri]))
                        results_all[cri+'_'+str(thr)][emb_name] = res_dict[cri]
                    AP_dsb[str(thr)].append(res.accuracy)
                print('#################')
                print('')

    results = {}
    for thr, acc in AP_dsb.items():
        results[thr] = np.mean(acc)
        
    with open(os.path.join(opbase, 'result.json'), 'w') as f:
        json.dump(results, f)

    with open(os.path.join(opbase, 'result_all.json'), 'w') as f:
        json.dump(results_all, f)
