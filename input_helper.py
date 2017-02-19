import os
import cv2


def input_process(input):
    # if it is a folder
    if os.path.isdir(input):
        file_list = os.listdir(input)
        file_list = [file for file in file_list if file.endswith('.jpg') or file.endswith('.png')]
        file_list.sort()
        img_list = []
        for file in file_list:
            img_list.append(os.path.join(input, file))
        return 1, img_list
    # it is an image
    elif input.endswith('.jpg') or input.endswith('.png'):
        img = cv2.imread(input)
        return 2, img
    # it is a video
    elif input.endswith('.mp4') or input.endswith('.ts') or input.endswith('.avi'):
        cap = cv2.VideoCapture(input)
        return 3, cap


def read_gt(path):
    with open(path, 'r') as f:
        data = f.readlines()
        l = []
        for line in data:
            b = line.split(',')
            b = list(map(float, b))
            x1 = int(min(b[0::2]))
            y1 = int(min(b[1::2]))
            x2 = int(max(b[0::2]))
            y2 = int(max(b[1::2]))
            l.append((x1,y1,x2,y2))
    return l


def read_vtb_gt(path):
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