import numpy as np
import cv2
import sys
import time

import kcftracker
from input_helper import input_process, read_gt

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01


# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)

            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True


def main(video, ground_truth=None):
    result = []
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
    global duration, inteval

    tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    draw_gt = False
    if (video == 0):
        cap = cv2.VideoCapture(0)
    else:

        if ground_truth:
            draw_gt = True
            gt_box = read_gt(ground_truth)
            initTracking = True
            l, t, r, b = gt_box[0]
            ix, iy, w, h = l, t, r - l, b - t

        inteval = 30
        ret, cap = input_process(video)
        if ret == 1:
            for fn, img_file in enumerate(cap):
                frame = cv2.imread(img_file)

                if (selectingObject):
                    cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
                elif (initTracking):
                    cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                    tracker.init([ix, iy, w, h], frame)

                    initTracking = False
                    onTracking = True
                elif (onTracking):
                    t0 = time.time()
                    boundingbox = tracker.update(frame)
                    t1 = time.time()

                    boundingbox = list(map(int, boundingbox))
                    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                                  (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255),
                                  2)
                    result.append((boundingbox[0], boundingbox[1], boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]))
                    duration = 0.8 * duration + 0.2 * (t1 - t0)
                    # duration = t1-t0
                    cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                if draw_gt:
                    x1, y1, x2, y2 = gt_box[fn]
                    # print(x1,y1,x2,y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cv2.imshow('tracking', frame)
                # time.sleep(0.05)
                c = cv2.waitKey(inteval) & 0xFF
                if c == 27 or c == ord('q'):
                    break
                elif c == 32:
                    while True:
                        ch0 = 0xFF & cv2.waitKey(5)
                        if ch0 == 32: break
                        if selectingObject:
                            # when drag the window run this code
                            vis_copy = frame.copy()
                            cv2.rectangle(vis_copy, (ix, iy), (cx, cy), (0, 255, 0), 2)
                            cv2.imshow('tracking', vis_copy)

        elif ret == 3:
            while (cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break

                if (selectingObject):
                    cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
                elif (initTracking):
                    cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                    tracker.init([ix, iy, w, h], frame)

                    initTracking = False
                    onTracking = True
                elif (onTracking):
                    t0 = time()
                    boundingbox = tracker.update(frame)
                    t1 = time()

                    boundingbox = list(map(int, boundingbox))
                    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                                  (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255),
                                  2)

                    duration = 0.8 * duration + 0.2 * (t1 - t0)
                    # duration = t1-t0
                    cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                cv2.imshow('tracking', frame)
                c = cv2.waitKey(inteval) & 0xFF
                if c == 27 or c == ord('q'):
                    break
                elif c == 32:
                    while True:
                        ch0 = 0xFF & cv2.waitKey(5)
                        if ch0 == 32: break
                        if selectingObject:
                            # when drag the window run this code
                            vis_copy = frame.copy()
                            cv2.rectangle(vis_copy, (ix, iy), (cx, cy), (0, 255, 0), 2)
                            cv2.imshow('tracking', vis_copy)

            cap.release()
            cv2.destroyAllWindows()

    return result

if __name__ == '__main__':
    video = sys.argv[1]
    if len(sys.argv) == 3:
        gt = sys.argv[2]
    else:
        gt = None
    main(video, gt)

