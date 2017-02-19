#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 10/17/16
import os
import cv2
import time
import numpy as np
from input_helper import input_process

video_src = "/home/zhr/tensorflow/visual_tracking/data/video/vot2015"

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


if __name__ == '__main__':
    # videos = [dir for dir in os.listdir(video_src) if os.path.isdir(os.path.join(video_src, dir))]
    # videos.sort()
    # for number, video in enumerate(videos):
    #     gts = read_gt(os.path.join(video_src, video, 'groundtruth.txt'))
    #     kcf = read_res(os.path.join(video_src, video, 'kcf0_groundtruth.txt'))
    #     kcf2 = read_res(os.path.join(video_src, video, 'kcf2_groundtruth.txt'))
    #     print("%d of %d" % (number, len(videos)))
    #     ret, img_list = input_process(os.path.join(video_src, video))
    #     if ret == 1:
    #         for fn, img_file in enumerate(img_list):
    #             frame = cv2.imread(img_file)
    #             x1, y1, x2, y2 = gts[fn]
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    #             x1, y1, x2, y2 = kcf[fn]
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    #             x1, y1, x2, y2 = kcf2[fn]
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #             cv2.imshow("image", frame)
    #             c = cv2.waitKey(0) & 0xFF
    #             if c == 27 or c == ord('q'):
    #                 break
    #             time.sleep(0.05)
    #         cv2.destroyAllWindows()

    # video_set = [ 'ball1', 'basketball', 'blanket', 'gymnastics1', 'fish1', 'fish3', 'helicopter','pedestrian2']
    video_set = ['gymnastics1']
    for number, video in enumerate(video_set):
        gts = read_gt(os.path.join(video_src, video, 'groundtruth.txt'))
        kcf = read_res(os.path.join(video_src, video, 'kcf_groundtruth.txt'))
        kcf2 = read_res(os.path.join(video_src, video, 'kcf2_groundtruth.txt'))
        result_path = os.path.join('result', video)
        if not os.path.exists(os.path.join('result', video)):
            os.mkdir(os.path.join('result', video))
        print("%d of %d" % (number, len(video_set)))
        ret, img_list = input_process(os.path.join(video_src, video))
        if ret == 1:
            for fn, img_file in enumerate(img_list):
                frame = cv2.imread(img_file)
                x1, y1, x2, y2 = gts[fn]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                x1, y1, x2, y2 = kcf[fn]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                x1, y1, x2, y2 = kcf2[fn]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, '#' + str(fn), (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 255), 2)
                # cv2.imshow("image", frame)
                c = cv2.waitKey(10) & 0xFF
                if c == 27 or c == ord('q'):
                    break
                # time.sleep(0.05)
                output = os.path.join(result_path, img_file.split('/')[-1])
                cv2.imwrite(output, frame)
            cv2.destroyAllWindows()