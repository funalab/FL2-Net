import os
import os.path as pt
import csv
import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser(description='python split_411.py')
ap.add_argument('--input', '-i', nargs='?', default='images/classification')
ap.add_argument('--output', '-o', nargs='?', default='datasets/split_list_411')
ap.add_argument('--field', '-f', type=int, default=7)
ap.add_argument('--emb', '-e', type=int, default=12)
ap.add_argument('--num_train', '-tr', type=int, default=8)
ap.add_argument('--num_test', '-te', type=int, default=2)
ap.add_argument('--num_val', '-va', type=int, default=2)
ap.add_argument('--timepoint', '-tp', type=int, default=506)
args = ap.parse_args()

num_field = args.field
num_emb = args.emb
num_train = args.num_train
num_test = args.num_test
num_validation = args.num_val
timepoint = args.timepoint

seed = 111

train_dir = args.output + '/train'
os.makedirs(train_dir, exist_ok=False)
test_dir = args.output + '/test'
os.makedirs(test_dir, exist_ok=False)
val_dir = args.output + '/validation'
os.makedirs(val_dir, exist_ok=False)

fl = open(args.output + '/split_list_411.txt', 'w')
tr = open(train_dir + '/dataset.txt', 'w')
te = open(test_dir + '/dataset.txt', 'w')
va = open(val_dir + '/dataset.txt', 'w')

for field in range(1, num_field+1):
    fl.write('F' + str(field).zfill(3))
    np.random.seed(seed)
    emb_rand = np.random.choice(num_emb, num_emb, replace=False)

    with open(args.input + '/' + 'F{}.csv'.format(str(field).zfill(3)), 'r') as f:
        reader = csv.reader(f)
        div_list = [row for row in reader]

        
    fl.write('\n[train]\n')
    emb_train = [n+1 for n in emb_rand[:num_train]]
    for emb in emb_train:
        for tp in range(1, timepoint+1):
            tr.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
            with open(train_dir + '/dataset0.txt', 'a') as div0, open(train_dir + '/dataset1.txt', 'a') as div1, open(train_dir + '/dataset2.txt', 'a') as div2:
                if tp < int(div_list[emb][1]):
                    div0.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                elif tp < int(div_list[emb][2]):
                    div1.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                else:
                    div2.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
    fl.writelines([str(n)+' ' for n in emb_train])

    
    fl.write('\n[test]\n')
    emb_test = [n+1 for n in emb_rand[num_train : num_train+num_test]]
    for emb in emb_test:
        for tp in range(1, timepoint+1):
            te.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
            with open(test_dir + '/dataset0.txt', 'a') as div0, open(test_dir + '/dataset1.txt', 'a') as div1, open(test_dir + '/dataset2.txt', 'a') as div2:
                if tp < int(div_list[emb][1]):
                    div0.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                elif tp < int(div_list[emb][2]):
                    div1.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                else:
                    div2.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
    fl.writelines([str(n)+' ' for n in emb_test])

    
    fl.write('\n[validation]\n')
    emb_val = [n+1 for n in emb_rand[num_train+num_test : num_train+num_test+num_validation]]
    for emb in emb_val:
        for tp in range(1, timepoint+1):
            va.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
            with open(val_dir + '/dataset0.txt', 'a') as div0, open(val_dir + '/dataset1.txt', 'a') as div1, open(val_dir + '/dataset2.txt', 'a') as div2:
                if tp < int(div_list[emb][1]):
                    div0.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                elif tp < int(div_list[emb][2]):
                    div1.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
                else:
                    div2.write('F' + str(field).zfill(3) + '/Embryo' + str(emb).zfill(2) + '/' + str(tp).zfill(3) + '.tif\n')
    fl.writelines([str(n)+' ' for n in emb_val])

    fl.write('\n\n')
    seed += 1
    
fl.close()
tr.close()
te.close()
va.close()
