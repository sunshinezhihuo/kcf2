import os
from run import main

video_src = "/home/zhr/tensorflow/visual_tracking/data/video/vot2015"


if __name__ == '__main__':

    videos = [dir for dir in os.listdir(video_src) if os.path.isdir(os.path.join(video_src, dir))]
    gts = []
    for video in videos:
        gts.append(os.path.join(video_src, video, 'groundtruth.txt'))

    for video, gt in zip(videos, gts):
        res = main(os.path.join(video_src, video), gt)
        res_path = os.path.join(video_src, video, 'kcf_groundtruth.txt')

        with open(res_path, 'w') as fp:
            for bbox in res:
                line = str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[2]) + '\n'
                fp.writelines(line)

