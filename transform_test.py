#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 10/14/16
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = "../data/video/vot2015/bag/00000001.jpg"
bbox = "291.827,203.603,396.382,264.289,442.173,185.397,337.618,124.711"

b = bbox.split(',')
b = list(map(float, b))
p1 = b[:2]
p2 = b[2:4]
p3 = b[4:6]
p4 = b[6:8]
r = np.array([p1,p2,p3,p4])
x = r[:, 0].argmin()
y = x + 4
if r[x,0] == r[(y-1)%4, 0]:
    x = (y-1)%4

r = np.array([r[x], r[(x+4+1)%4], r[(x+4+2)%4], r[(x+4+3)%4]])
if r[0,0] == r[1, 0]:
    r = np.array([r[1], r[0], r[3], r[2]])


if __name__ == '__main__':
    img = cv2.imread(img_path,flags=cv2.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]

    cx = r[:, 0].sum() / 4
    cy = r[:, 1].sum() / 4
    w = ((r[3][0] - r[0][0]) ** 2 + (r[3][1] - r[0][1]) ** 2) ** 0.5
    h = ((r[1][0] - r[0][0]) ** 2 + (r[1][1] - r[0][1]) ** 2) ** 0.5

    cv2.circle(img, (int(cx),int(cy)), 4, (255,0,0), 2)
    box = r.reshape(-1, 1, 2)
    box = box.astype(np.int32)
    img = cv2.polylines(img, [box], True, (255, 0, 0), 3, lineType=cv2.LINE_4)

    dx, dy = cols/2 - cx, rows/2 - cy
    shift = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img, shift, (cols, rows))

    init_angle = np.arctan((r[1][0]-r[0][0])/(r[1][1]-r[0][1]))
    init_angle = init_angle*180/np.pi
    rotate = cv2.getRotationMatrix2D((cols/2,rows/2), -init_angle, 1)
    dst2 = cv2.warpAffine(dst, rotate, (cols, rows))

    rotate2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 1, 1)
    dst3 = cv2.warpAffine(dst2, rotate2, (cols, rows))
    shift2 = np.float32([[1, 0, 3], [0, 1, 2]])
    dst3 = cv2.warpAffine(dst3, shift2, (cols, rows))

    angle = -init_angle + 1
    cx_ = cols/2 + 2
    cy_ = rows/2 - 3
    w_ = w*1.05
    h_ = h
    cc = np.array([cx_, cy_, 1])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    M = M.reshape(-1).tolist() + [0,0,1]
    M = np.array(M).reshape(-1,3,3)
    cc = np.dot(np.linalg.inv(M), cc)

    shift_ = np.array(shift.reshape(-1).tolist() + [0,0,1]).reshape(-1,3,3)
    cc = np.dot(np.linalg.inv(shift_), cc[0])

    img_ = img.copy()
    img_ = cv2.circle(img_, (int(cc[0][0]), int(cc[0][1])), 4, (255, 0, 0), 2)

    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(dst)
    plt.subplot(223)
    plt.imshow(dst2)
    plt.subplot(224)
    plt.imshow(img_)

    plt.show()