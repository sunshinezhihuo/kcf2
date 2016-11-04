import cv2
import sys
import time
import os
import kcftracker
import numpy as np
from matplotlib import pyplot as plt
from input_helper import input_process, read_gt

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

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

    tracker = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    dirname, window_name = os.path.split(video)
    if window_name == '':
        window_name = os.path.basename(dirname)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_boundingbox)

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

            dirname = os.path.dirname(ground_truth)
            kcf_name = os.path.join(dirname, "kcf_groundtruth.txt")
            kcf_box = read_gt(kcf_name)

        inteval = 10
        peak = []
        ret, cap = input_process(video)
        if ret == 1:
            for fn, img_file in enumerate(cap):
                frame = cv2.imread(img_file)
                rows, cols = frame.shape[:2]

                if (selectingObject):
                    cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
                elif (initTracking):
                    tracker.init([ix, iy, w, h], frame)
                    result.append((ix,iy,ix+w,iy+h))
                    cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                    initTracking = False
                    onTracking = True
                elif (onTracking):
                    t0 = time.time()
                    boundingbox, peak_value = tracker.update(frame)
                    peak.append(peak_value)
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
                    cv2.putText(frame, "peak: " + str(peak_value), (8, rows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)

                if draw_gt:
                    x1, y1, x2, y2 = gt_box[fn]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    x1, y1, x2, y2 = kcf_box[fn]
                    # print(x1,y1,x2,y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, "num: " + str(fn), (8, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                cv2.imshow(window_name, frame)
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
                            cv2.imshow(window_name, vis_copy)

            cv2.destroyAllWindows()
            x = np.linspace(0, len(peak)-1, len(peak))
            peak = np.array(peak)
            plt.plot(x, peak)
            plt.grid()
            # plt.show()

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
                    t0 = time.time()
                    boundingbox, peak_value = tracker.update(frame)
                    t1 = time.time()

                    boundingbox = list(map(int, boundingbox))
                    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                                  (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255),
                                  2)

                    duration = 0.8 * duration + 0.2 * (t1 - t0)
                    # duration = t1-t0
                    cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                cv2.imshow(window_name, frame)
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
                            cv2.imshow(window_name, vis_copy)

            cap.release()
            cv2.destroyAllWindows()

    return result

if __name__ == '__main__':
    video = sys.argv[1]
    if len(sys.argv) == 3:
        gt = sys.argv[2]
    else:
        gt = None
    res = main(video, gt)

    # res_path = os.path.join(video, 'kcf2_groundtruth.txt')
    # with open(res_path, 'w') as fp:
    #     for bbox in res:
    #         line = str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + '\n'
    #         fp.writelines(line)
