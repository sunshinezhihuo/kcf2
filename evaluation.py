import os
from run import main

video_src = "/home/zhr/tensorflow/visual_tracking/data/video/vot2015"


if __name__ == '__main__':

    videos = [dir for dir in os.listdir(video_src) if os.path.isdir(os.path.join(video_src, dir))]
    videos.sort()
    gts = []
    for video in videos:
        gts.append(os.path.join(video_src, video, 'groundtruth.txt'))

    for i,(video, gt) in enumerate(zip(videos, gts)):
        res = main(os.path.join(video_src, video), gt)
        res_path = os.path.join(video_src, video, 'kcf0_groundtruth.txt')
        print("%d th of %d videos" % (i, len(videos)))
        with open(res_path, 'w') as fp:
            for bbox in res:
                line = str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + '\n'
                fp.writelines(line)

