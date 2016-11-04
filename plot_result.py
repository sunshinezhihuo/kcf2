#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 10/16/16
from matplotlib import pyplot as plt
import numpy as np
import os

dataset = "/home/zhr/tensorflow/visual_tracking/data/video/vot2015"


def read_gt(path):
    gt_box = []
    with open(path, 'r') as f:
        data = f.readlines()
        for line in data:
            b = line.split(',')
            b = list(map(float, b))
            x1 = int(min(b[0::2]))
            y1 = int(min(b[1::2]))
            x2 = int(max(b[0::2]))
            y2 = int(max(b[1::2]))
            gt_box.append((x1, y1, x2, y2))
    return gt_box


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
    for key, line in res_lines.items():
        p = ax.plot(x, line, label=key)
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels)
    # plt.legend(plot_handler, plot_name, 'best', numpoints=1)
    plt.show()

    standard = 0.5
    arg = x.tolist().index(standard)
    result = dict()
    for key, line in res_lines.items():
        result[key] = line[arg]
    return result


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
    for key, line in res_lines.items():
        p = ax.plot(x, line, label=key)
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels)
    # plt.legend(plot_handler, plot_name, 'best', numpoints=1)
    plt.show()

    standard = 20
    arg = x.tolist().index(standard)
    result = dict()
    for key, line in res_lines.items():
        result[key] = line[arg]
    return result


if __name__ == '__main__':
    videos = [dir for dir in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, dir))]
    gt_boxs = dict()
    res0_boxs = dict()
    res1_boxs = dict()
    res2_boxs = dict()
    for video in videos:
        gt_path = os.path.join(dataset, video, 'groundtruth.txt')
        res0_path = os.path.join(dataset, video, 'kcf0_groundtruth.txt')
        res1_path = os.path.join(dataset, video, 'kcf_groundtruth.txt')
        res2_path = os.path.join(dataset, video, 'kcf2_groundtruth.txt')
        gt = read_gt(gt_path)
        res0 = read_res(res0_path)
        res1 = read_res(res1_path)
        res2 = read_res(res2_path)
        gt_boxs[video] = gt
        res0_boxs[video] = res0
        res1_boxs[video] = res1
        res2_boxs[video] = res2
    res_box = {"kcf0":res0_boxs, "kcf":res1_boxs, "kcf2":res2_boxs}

    res = successPlot_all(gt_boxs, res_box)
    print(res)
    res = precisionPlot_all(gt_boxs, res_box)
    print(res)
    # import  sys
    # video_src = sys.argv[1]
    # gt_path = os.path.join(video_src, "groundtruth.txt")
    # kcf_path = os.path.join(video_src, "kcf_groundtruth.txt")
    # kcf2_path = os.path.join(video_src, "kcf2_groundtruth.txt")
    # successPlot(read_gt(gt_path), read_res(kcf_path), read_res(kcf2_path))
