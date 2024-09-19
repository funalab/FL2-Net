import numpy as np
import os
import sys
from argparse import ArgumentParser

if __name__ == '__main__':
    ap = ArgumentParser(description='python dataset_sampling.py')
    ap.add_argument('--inlist', '-il', nargs='?', default='datasets/split_list_411/validation.txt', help='Specify split list for validation data')
    ap.add_argument('--outlist', '-ol', nargs='?', default='datasets/split_list_411/validation_sample.txt', help='Specify sampled split list for validation data')
    ap.add_argument('--t_from', '-f', type=int, default=1, help='first timepoint')
    ap.add_argument('--t_to', '-t', type=int, default=506, help='final timepoint')
    ap.add_argument('--interval', '-i', type=int, default=50, help='interval of sampling')
    args = ap.parse_args()
    argvs = sys.argv

    with open(args.inlist, 'r') as f_in, open(args.outlist, 'w') as f_out:
        dlist = f_in.read().split()
        dlist.sort()
        for d in dlist:
            emb_name = os.path.dirname(d)
            tp = int(os.path.splitext(os.path.basename(d))[0])
            if (tp - args.t_from) % args.interval != 0:
                continue
            elif args.t_to < tp:
                continue
            else:
                f_out.write(d)
                f_out.write('\n')
