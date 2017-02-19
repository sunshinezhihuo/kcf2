#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 10/16/16
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
import os




def read_gt(path):
    with open(path, 'r') as f:
        data = f.readlines()
        l = []
        for line in data:
            b = line.split(',')
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(b[0]) + int(b[2])
            y2 = int(b[1]) + int(b[3])
            l.append((x1,y1,x2,y2))
    return l


def read_res(path):
    res_box = []
    with open(path, 'r') as f:
        data = f.readlines()
        for line in data:
            b = line.split(',')
            b = list(map(float, b))
            res_box.append(np.array(b).astype(np.int16))
    return res_box


def successPlot(gt_box, res_box, res2_box):
    step = 100
    filename = 'success.png'
    plt.ylabel("Success")
    plt.xlabel("Overlap threshold")
    plt.title("Overall Success")
    # plt.axis([0, 1, 0, 1])
    plt.grid()

    x = np.linspace(0, 1, step + 1)
    res1_line = []
    res2_line = []
    datalen = len(gt_box)

    iou_list = []
    for i in range(1, datalen):
        x11, y11, x12, y12 = gt_box[i]
        x21, y21, x22, y22 = res_box[i]
        x_list = [x11, x12, x21, x22]
        x_list.sort()
        if x_list[:2] == [x11, x12] or x_list[:2] == [x21, x22]:
            continue
        else:
            deta_x = x_list[2] - x_list[1]

        y_list = [y11, y12, y21, y22]
        y_list.sort()
        if y_list[:2] == [y11, y12] or y_list[:2] == [y21, y22]:
            continue
        else:
            deta_y = y_list[2] - y_list[1]

        s1 = np.uint32(deta_x) * np.uint32(deta_y)
        s2 = np.uint32(x12 - x11) * np.uint32(y12 - y11) + np.uint32(x22 - x21) * np.uint32(y22 - y21) - s1
        iou_list.append(s1 / s2)

    for iou in x:
        if iou == 0:
            res1_line.append(1.0)
            continue
        positive = 0
        for ii in iou_list:
            if ii >= iou: positive += 1
        res1_line.append(positive / datalen)

    iou_list = []
    for i in range(1, datalen):
        x11, y11, x12, y12 = gt_box[i]
        x21, y21, x22, y22 = res2_box[i]
        x_list = [x11, x12, x21, x22]
        x_list.sort()
        if x_list[:2] == [x11, x12] or x_list[:2] == [x21, x22]:
            continue
        else:
            deta_x = x_list[2] - x_list[1]

        y_list = [y11, y12, y21, y22]
        y_list.sort()
        if y_list[:2] == [y11, y12] or y_list[:2] == [y21, y22]:
            continue
        else:
            deta_y = y_list[2] - y_list[1]

        s1 = np.uint32(deta_x) * np.uint32(deta_y)
        s2 = np.uint32(x12 - x11) * np.uint32(y12 - y11) + np.uint32(x22 - x21) * np.uint32(y22 - y21) - s1
        iou_list.append(s1 / s2)

    for iou in x:
        if iou == 0:
            res2_line.append(1.0)
            continue
        positive = 0
        for ii in iou_list:
            if ii >= iou: positive += 1
        res2_line.append(positive / datalen)

    ax = plt.subplot(1, 1, 1)
    ax.plot(x, res1_line, label="kcf")
    ax.plot(x, res2_line, label="kcf2")
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels)
    plt.show()

def successPlot_all(gt_boxs, res_boxs):
    total_point = 100
    filename = 'success.png'
    plt.ylabel("Success")
    plt.xlabel("Overlap threshold")
    plt.title("Overall Success")
    # plt.axis([0, 1, 0, 1])
    plt.grid()

    x = np.linspace(0, 1, total_point + 1)

    all_box_len = 0
    for key, item in gt_boxs.items():
        all_box_len += len(item) - 1

    res_lines = dict()

    for name, result in res_boxs.items():
        res_lines[name] = []

        ilist = dict()
        for video_name, bbox in result.items():
            iou_list = []
            for index in range(1, len(bbox)):
                x11, y11, x12, y12 = gt_boxs[video_name][index]
                x21, y21, x22, y22 = bbox[index]
                x_list = [x11, x12, x21, x22]
                x_list.sort()
                if x_list[:2] == [x11, x12] or x_list[:2] == [x21, x22]:
                    iou_list.append(0)
                    continue
                else:
                    deta_x = x_list[2] - x_list[1]

                y_list = [y11, y12, y21, y22]
                y_list.sort()
                if y_list[:2] == [y11, y12] or y_list[:2] == [y21, y22]:
                    iou_list.append(0)
                    continue
                else:
                    deta_y = y_list[2] - y_list[1]

                s1 = np.uint32(deta_x) * np.uint32(deta_y)
                s2 = np.uint32(x12 - x11) * np.uint32(y12 - y11) + np.uint32(x22 - x21) * np.uint32(y22 - y21) - s1
                iou_list.append(s1 / s2)
                # print("s1: ",s1, "s2:", s2)
            ilist[video_name] = iou_list

        for iou in x:
            if iou == 0:
                res_lines[name].append(1.0)
                continue

            positive = 0
            for _, l in ilist.items():
                for s in l:
                    if s >= iou: positive += 1
            res_lines[name].append(positive/all_box_len)
        res_lines[name][0] = res_lines[name][1]

    ax = plt.subplot(1, 1, 1)
    standard = 0.5
    arg = x.tolist().index(standard)
    orde = []
    for key, line in res_lines.items():
        pp = line[arg]
        orde.append(pp)
        if key == 'ours':
            p = ax.plot(x, line, label=key + ' [' + '%.3f' % pp + ']', linewidth=3)
        else:
            p = ax.plot(x, line, label=key + ' [' + '%.3f' % pp + ']')
    handles, labels = ax.get_legend_handles_labels()

    orde2 = sorted(orde, reverse=True)
    handles2 = []
    labels2 = []
    for o in orde2:
        n = orde.index(o)
        handles2.append(handles[n])
        labels2.append(labels[n])

    ax.legend(handles2, labels2)
    # plt.legend(plot_handler, plot_name, 'best', numpoints=1)
    plt.show()
    return None


def precisionPlot_all(gt_boxs, res_boxs):
    thrMax = 50
    total_point = 100
    filename = 'precision.png'
    plt.ylabel("Precision")
    plt.xlabel("Location error threshold")
    plt.title("Overall Precision")
    # plt.axis([0, thrMax, 0, 1])
    plt.grid()
    x = np.linspace(0, thrMax, total_point + 1)

    all_box_len = 0
    for key, item in gt_boxs.items():
        all_box_len += len(item) - 1

    res_lines = dict()
    for name, result in res_boxs.items():
        res_lines[name] = []

        ilist = dict()
        for video_name, bbox in result.items():
            iou_list = []
            for index in range(1, len(bbox)):
                x11, y11, x12, y12 = gt_boxs[video_name][index]
                x21, y21, x22, y22 = bbox[index]
                c1 = (x12+x11)/2, (y11+y12)/2
                c2 = (x21+x22)/2, (y21+y22)/2
                d = (np.uint32((c1[0] - c2[0])**2) + np.uint32((c1[1] - c2[1])**2))**0.5
                iou_list.append(d)
                # print("s1: ",s1, "s2:", s2)
            ilist[video_name] = iou_list

        for iou in x:
            if iou == 0:
                res_lines[name].append(0)
                continue

            positive = 0
            for _, l in ilist.items():
                for s in l:
                    if s <= iou: positive += 1
            res_lines[name].append(positive/all_box_len)
        # res_lines[name][0] = res_lines[name][1]

    ax = plt.subplot(1, 1, 1)
    standard = 20
    arg = x.tolist().index(standard)
    orde = []
    for key, line in res_lines.items():
        pp = line[arg]
        orde.append(pp)
        if key == 'ours':
            p = ax.plot(x, line, label=key + ' [' + '%.3f' % pp + ']', linewidth=3)
        else:
            p = ax.plot(x, line, label=key + ' [' + '%.3f' % pp + ']')
    handles, labels = ax.get_legend_handles_labels()

    orde2 = sorted(orde, reverse=True)
    handles2 = []
    labels2 = []
    for o in orde2:
        n = orde.index(o)
        handles2.append(handles[n])
        labels2.append(labels[n])

    ax.legend(handles2, labels2)
    plt.show()
    return None


if __name__ == '__main__':
    dataset = "/home/zhr/tensorflow/visual_tracking/data/video/CVPR13"

    algo = ["CSK", "TLD", "CT", "Struck", "KMS", "VTD", "OAB", "DFT", "MIL"] # , "SCM"
    algo_path = "/home/zhr/tensorflow/visual_tracking/data/video/VTB50_results/results/results_TRE_CVPR13"

    videos = [dir for dir in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, dir))]
    videos.sort()
    gt_boxs = dict()
    res0_boxs = dict()
    others_boxs = dict()
    for al in algo:
        others_boxs[al] = dict()

    for video in videos:
        gt_path = os.path.join(dataset, video, 'groundtruth_rect.txt')
        gt = read_gt(gt_path)
        gt_boxs[video] = gt

        res0_path = os.path.join(dataset, video, 'kcf2_groundtruth.txt')
        res0 = read_res(res0_path)
        res0_boxs[video] = res0

        for al in algo:
            file_path = os.path.join(algo_path, video[0].lower() + video[1:] +'_'+al+'.mat')
            data = scipy.io.loadmat(file_path)
            tmp = data['results'][0][0][0][0]
            if len(tmp[0]) > 10:
                bounding_box = tmp[0]
            else:
                bounding_box = tmp[1]
            bounding_box[:,2] += bounding_box[:,0]
            bounding_box[:,3] += bounding_box[:,1]
            others_boxs[al][video] = bounding_box
    res_box = {"ours":res0_boxs}
    res_box.update(others_boxs)

    res = successPlot_all(gt_boxs, res_box)
    res = precisionPlot_all(gt_boxs, res_box)
