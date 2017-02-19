#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 11/4/16
import numpy as np
import cv2
import sys
import scipy.io
from numba import jit
from matplotlib import pyplot as plt

# @jit(cache=True)
def im2color(img, color, sizeX=None, sizeY=None):
    w2c_path = 'w2c.mat'
    probmap = scipy.io.loadmat(w2c_path)['w2c']

    # order of colornames: black, blue, brown, grey, green, orange, pink, purple, red, white, yellow
    color_values = np.array([[0, 0, 0], [0, 0, 1], [.5, .4, .25], [.5, .5, .5], [0, 1, 0], [1, .8, 0],
                             [1, .5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
    BB, GG, RR = cv2.split(img.astype(np.double))

    index_im = (RR / 8).astype(np.int16) + 32 * ((GG / 8).astype(np.int16)) + 32 * 32 * ((BB / 8).astype(np.int16))

    if color == 0:
        argmax = probmap.argmax(1)
        argmax = np.array(argmax)
        outmap = argmax[index_im]
        return outmap

    elif color == -1:
        argmax = probmap.argmax(1)
        argmax = np.array(argmax)
        outmap = argmax[index_im]
        colormap = (color_values[outmap] * 255).astype(np.uint8)
        return colormap

    elif color == 1:
        probalistic = probmap[index_im]
        cell_size = 4
        cy, cx = int(img.shape[0]/2), int(img.shape[1]/2)
        x1, y1, x2, y2 = cx-cell_size*sizeX*0.5, cy-cell_size*sizeY*0.5, cx+cell_size*sizeX*0.5, cy+cell_size*sizeY*0.5
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_clip = probalistic[y1:y2, x1:x2]
        img_cell = []
        for y in range(sizeY):
            for x in range(sizeX):
                tmp = img_clip[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size].reshape(-1,11)
                img_cell.append(tmp)
        img_cell = np.array(img_cell)
        img_cell = img_cell.sum(1) / cell_size*cell_size
        img_cell = img_cell.transpose(1, 0)
        return img_cell


if __name__ == '__main__':
    w2c_path = 'w2c.mat'
    img_file = sys.argv[1]
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    map = im2color(img_file, -1)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(map)
    plt.show()
