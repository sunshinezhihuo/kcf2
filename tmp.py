#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 11/20/16

path = "/home/zhr/tensorflow/visual_tracking/data/video/VTB50_results/results/results_TRE_CVPR13"

if __name__ == '__main__':
    import scipy.io
    import os
    file = os.path.join(path, 'david_SCM.mat')
    if os.path.exists(file):
        data = scipy.io.loadmat(file)
        r = data['results']
        rect = r[0][0][0][0][-1]
        print(len(rect))
    pass
    pass
