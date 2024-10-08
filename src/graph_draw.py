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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import skimage.io as skimage
from skimage import transform as tr
import skimage.morphology as mor
from argparse import ArgumentParser
from datetime import datetime
import pytz
plt.style.use('ggplot')

class GraphDraw():

    def __init__(self, opbase, file_list):
        self.save_dir = save_dir
        # self.scale = 0.8 * 0.8 * 2.0
        self.scale = 1.0 * 1.0 * 1.0
        self.fn = file_list
        self.density = 0
        self.roi_pixel_num = 0


    def graph_draw_number(self, Time, Count):
        # Count
        cmap =  plt.get_cmap('Paired')
        label = []
        plt.figure()
        for num in range(len(Count)):
            colors = cmap(float(num) / len(Count))
            plt.plot(Time[:len(Count[num])], Count[num], color=colors, alpha=0.8, linewidth=1.0)
            label.append(self.fn[num])
            #label.append(self.fn[num][self.fn[num].find('E'):self.fn[num].find('E')+15])
            #if np.max(Count[num][:216]) >= 10:
            if np.max(Count[num][:100]) >= 5:
                print(self.fn[num])
                print(np.argmax(Count[num][:100]))
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Number of Nuclei', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'number.pdf'))

        plt.figure(figsize=(4, 14))
        for num in range(len(Count)):
            colors = cmap(float(num) / len(Count))
            plt.plot(1, 1, label=label[num], color=colors)
        plt.legend(label, loc=1)
        plt.savefig(os.path.join(self.save_dir, 'number_legend.pdf'))


    def graph_draw_synchronicity(self, Time, Count):
        # Count
        cell_stage = [2, 3, 4, 5, 8, 9, 16, 17, 32, 128]
        label = ['2-cell stage', '3-cell stage', '4-cell stage', '5- to 7-cell stage', '8-cell stage', '9- to 15-cell stage', '16-cell stage', '17- to 31-cell stage', '32-cell stage', '33- to 63-cell stage', '64-cell stage', '65- to 127-cell stage', '128-cell stage or more']
        all_period_cell = []
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(Count)):
            period_cell = []
            current_state = 1
            consist_flag = 0
            for tp in range(len(Count[num])):
                if Count[num][tp] >= cell_stage[current_state]:
                    consist_flag += 1
                    if consist_flag > 5:
                        current_state += 1
                        period_cell.append(Time[tp])
                        consist_flag = 0
                else:
                    consist_flag = 0
            period_cell.append(Time[tp])
            ''' for legend '''
            for i in range(len(period_cell)):
                colors = cmap(i+1)
                plt.barh(len(Count) - num, period_cell[i], height=0.0001, align='center', color=colors)
            ''' for plot '''
            for i in range(len(period_cell)):
                colors = cmap(len(period_cell) - i)
                plt.barh(len(Count) - num, period_cell[-(i+1)], height=0.8, align='center', label=label[len(period_cell) - (i+1)], color=colors)
        plt.yticks(range(1, len(Count)+1), ["" for i in range(0, len(Count))])
        plt.xlabel('Time [day]', size=12)
        plt.xlim([0.0, 3.5])
        plt.savefig(os.path.join(self.save_dir, 'cell_division_synchronization.pdf'))

        plt.figure()
        for i in range(len(period_cell)):
            colors = cmap(i+1)
            plt.plot(1, 1, color=colors)
        plt.legend(label, loc=1)
        plt.savefig(os.path.join(self.save_dir, 'cell_division_synchronization_legend.pdf'))


    def graph_draw_volume(self, Time, MeanVol, StdVol):
        # Volume Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanVol)):
            colors = cmap(float(num) / len(MeanVol))
            plt.plot(Time[:len(MeanVol[num])], np.array(MeanVol[num]) * self.scale, color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Volume [$\mu m^{3}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'volume_mean.pdf'))

        plt.figure()
        for num in range(len(StdVol)):
            colors = cmap(float(num) / len(StdVol))
            plt.plot(Time[:len(StdVol[num])], np.array(StdVol[num]) * self.scale, color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Volume (standard deviation) [$\mu m^{3}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'volume_std.pdf'))


    def graph_draw_surface(self, Time, MeanArea, StdArea):
        # Surface Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanArea)):
            colors = cmap(float(num) / len(MeanArea))
            plt.plot(Time[:len(MeanArea[num])], np.array(MeanArea[num]) * self.scale, color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Surface Area [$\mu m^{2}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'surface_area_mean.pdf'))

        plt.figure()
        for num in range(len(MeanArea)):
            colors = cmap(float(num) / len(MeanArea))
            plt.plot(Time[:len(StdArea[num])], np.array(StdArea[num]) * self.scale, color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Surface Area (standard deviation) [$\mu m^{2}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'surface_area_std.pdf'))


    def graph_draw_surface_volume(self, Time, MeanArea, StdArea, MeanVol, StdVol):
        assert len(MeanArea) == len(MeanVol)

        # Surface Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanArea)):
            colors = cmap(float(num) / len(MeanArea))
            plt.plot(Time[:len(MeanArea[num])], np.array(MeanArea[num]) / np.array(MeanVol[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Specific Surface Area [$\mu m^{-1}$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'surface_volume_ratio_mean.pdf'))


    def graph_draw_aspect_ratio(self, Time, MeanAsp, StdAsp):
        # Aspect Ratio Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanAsp)):
            colors = cmap(float(num) / len(MeanAsp))
            plt.plot(Time[:len(MeanAsp[num])], np.array(MeanAsp[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Aspect Ratio', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'aspect_ratio_mean.pdf'))

        plt.figure()
        for num in range(len(StdAsp)):
            colors = cmap(float(num) / len(StdAsp))
            plt.plot(Time[:len(StdAsp[num])], np.array(StdAsp[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Aspect Ratio (standard deviation)', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'aspect_ratio_std.pdf'))


    def graph_draw_solidity(self, Time, MeanSol, StdSol):
        # Solidity Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanSol)):
            colors = cmap(float(num) / len(MeanSol))
            plt.plot(Time[:len(MeanSol[num])], np.array(MeanSol[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Solidity', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'solidity_mean.pdf'))

        plt.figure()
        for num in range(len(StdSol)):
            colors = cmap(float(num) / len(StdSol))
            plt.plot(Time[:len(StdSol[num])], np.array(StdSol[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Solidity (standard deviation)', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'solidity_std.pdf'))


    def graph_draw_centroid(self, Time, MeanCen, StdCen):
        # Centroid Mean & SD
        cmap =  plt.get_cmap('Paired')
        plt.figure()
        for num in range(len(MeanCen)):
            colors = cmap(float(num) / len(MeanCen))
            plt.plot(Time[:len(MeanCen[num])], np.array(MeanCen[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Centroid [$\mu m$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'centroid_mean.pdf'))

        plt.figure()
        for num in range(len(StdCen)):
            colors = cmap(float(num) / len(StdCen))
            plt.plot(Time[:len(StdCen[num])], np.array(StdCen[num]), color=colors, alpha=0.8, linewidth=1.0)
        plt.xlabel('Time [day]', size=12)
        plt.ylabel('Centroid (standard deviation) [$\mu m$]', size=12)
        if Time[-1] != 0:
            plt.xlim([0.0, round(Time[-1], 1)])
        plt.savefig(os.path.join(self.save_dir, 'centroid_std.pdf'))

if __name__ == '__main__':
    ap = ArgumentParser(description='python graph_draw.py')
    ap.add_argument('--root', '-r', nargs='?', default='embryo_classification/datasets', help='Specify root path')
    ap.add_argument('--save_dir', '-o', nargs='?', default='results/figures_criteria', help='Specify output files directory for create figures')
    ap.add_argument('--list', '-l', nargs='?', default=None, help='Specify list file to draw graph')
    args = ap.parse_args()
    argvs = sys.argv

    # Make Directory
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = '{0}_{1}'.format(args.save_dir, current_datetime)
    os.makedirs(save_dir, exist_ok=True)

    if args.list == None:
        file_list = os.listdir(args.root)
    else:
        with open(args.list, 'r') as f:
            file_list = f.read().split()
    file_list.sort()

    number = []
    volume_mean, volume_sd = [], []
    surface_mean, surface_sd = [], []
    aspect_ratio_mean, aspect_ratio_sd = [], []
    solidity_mean, solidity_sd = [], []
    centroid_mean, centroid_sd = [], []
    for fl in file_list:
        file_name = os.path.join(args.root, fl, 'criteria.json')
        print('read: {}'.format(file_name))
        with open(file_name, 'r') as f:
            criteria_value = json.load(f)
        criteria_list = criteria_value.keys()

        if 'number' in criteria_list:
            number.append(criteria_value['number'])
        if 'volume_mean' in criteria_list:
            volume_mean.append(criteria_value['volume_mean'])
        if 'volume_sd' in criteria_list:
            volume_sd.append(criteria_value['volume_sd'])
        if 'surface_mean' in criteria_list:
            surface_mean.append(criteria_value['surface_mean'])
        if 'surface_sd' in criteria_list:
            surface_sd.append(criteria_value['surface_sd'])
        if 'aspect_ratio_mean' in criteria_list:
            aspect_ratio_mean.append(criteria_value['aspect_ratio_mean'])
        if 'aspect_ratio_sd' in criteria_list:
            aspect_ratio_sd.append(criteria_value['aspect_ratio_sd'])
        if 'solidity_mean' in criteria_list:
            solidity_mean.append(criteria_value['solidity_mean'])
        if 'solidity_sd' in criteria_list:
            solidity_sd.append(criteria_value['solidity_sd'])
        if 'centroid_mean' in criteria_list:
            centroid_mean.append(criteria_value['centroid_mean'])
        if 'centroid_sd' in criteria_list:
            centroid_sd.append(criteria_value['centroid_sd'])

    # Time Scale
    dt = 10 / float(60 * 24)
    count_max = 0
    for i in range(len(number)):
        count_max = np.max([len(number[i]), count_max])
    time_point = [dt * x for x in range(count_max)]

    gd = GraphDraw(save_dir, file_list)
    gd.graph_draw_number(time_point, number)
    gd.graph_draw_synchronicity(time_point, number)
    gd.graph_draw_volume(time_point, volume_mean, volume_sd)
    gd.graph_draw_surface(time_point, surface_mean, surface_sd)
    gd.graph_draw_surface_volume(time_point, volume_mean, volume_sd, surface_mean, surface_sd)
    gd.graph_draw_aspect_ratio(time_point, aspect_ratio_mean, aspect_ratio_sd)
    gd.graph_draw_solidity(time_point, solidity_mean, solidity_sd)
    gd.graph_draw_centroid(time_point, centroid_mean, centroid_sd)
