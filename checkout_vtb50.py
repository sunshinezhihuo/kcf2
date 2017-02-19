#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 10/17/16
import os
import cv2
import time
import numpy as np
from input_helper import input_process
import scipy.io

video_src = "/home/zhr/tensorflow/visual_tracking/data/video/CVPR13"

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


def read_other(path):
    data = scipy.io.loadmat(path)


if __name__ == '__main__':
    videos = [dir for dir in os.listdir(video_src) if os.path.isdir(os.path.join(video_src, dir))]
    videos.sort()

    algo = ["CSK", "TLD", "CT", "Struck"]
    color = np.random.randint(0, 255, (len(algo), 3))
    algo_path = "/home/zhr/tensorflow/visual_tracking/data/video/VTB50_results/results/results_TRE_CVPR13"
    for number, video in enumerate(videos):
        gts = read_gt(os.path.join(video_src, video, 'groundtruth_rect.txt'))
        kcf = read_res(os.path.join(video_src, video, 'kcf2_groundtruth.txt'))
        algo_baseline = []
        for name in algo:
            file_path = os.path.join(algo_path, video[0].lower() + video[1:] +'_'+name+'.mat')
            data = scipy.io.loadmat(file_path)
            tmp = data['results'][0][0][0][0]
            # print(len(tmp))
            # for i in range(len(tmp)):
            #     print(tmp[i])
            if len(tmp[0]) > 10:
                algo_baseline.append(tmp[0])
            else:
                algo_baseline.append(tmp[1])


        print("%d of %d" % (number, len(videos)))
        ret, img_list = input_process(os.path.join(video_src, video, 'img'))
        if ret == 1:
            for fn, img_file in enumerate(img_list):
                frame = cv2.imread(img_file)
                x1, y1, x2, y2 = gts[fn]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, 'gt', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                x1, y1, x2, y2 = kcf[fn]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, 'ours', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                for n, base in enumerate(algo_baseline):
                    aa = base[fn]
                    x1, y1, x2, y2 = np.array([aa[0], aa[1], aa[0]+aa[2], aa[1]+aa[3]]).astype(np.uint16)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color[n].tolist(), 2)
                    cv2.putText(frame, algo[n], (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[n].tolist(), 1)

                cv2.imshow("image", frame)
                c = cv2.waitKey(10) & 0xFF
                if c == 27 or c == ord('q'):
                    break
                time.sleep(0.05)
            cv2.destroyAllWindows()

