import sys
import os
from argparse import ArgumentParser

import cv2
import time
import logging as log
import numpy as np
import simpleaudio
import sys
from PIL import Image
join=os.path.join

def get_pose(args):
    from music_main_nn_over_BS import Music


    sys.path.append('~/openpose/build/python');
    
    from openpose import pyopenpose as op

    params = dict()
    params["model_folder"] = "~/openpose/models/"
 
    
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()

    print("openpose has been imported")

    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                  [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                  [0, 14], [0, 15], [14, 16], [15, 17]]


    render_time = 0

    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    count = 0

    music_obj = Music(args.nn_frame_start, args.music_freq)
    sec = 0
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    
    notes = []
    poses = []
    count = 0
    while cap.isOpened():
        count += 1
        _, frame = cap.read()

        # frame = frame[:,:,::-1]
        frame = cv2.resize(frame, (384, 384))
        datum.cvInputData = frame
        # datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        _ = datum.cvOutputData
        pose = np.reshape(datum.poseKeypoints[0][:,:2], [-1])
        pose = np.array(list(pose[:16]) + list(pose[18:38]))

        points_return = []
        points = []

        for i in range(len(pose)//2):
            x = pose[2*i]
            y = pose[2*i+1]
            if (int(x) == 0 and int(y) == 0):
                points.append(None)
            else:
                add_keypoints(frame, x, y, i)
                points.append((int(x), int(y)))
            points_return += normalize(int(x), int(y), frame.shape[1], frame.shape[0])

        prev_frame_time = draw_skeleton(POSE_PAIRS, frame, points, prev_frame_time, args.wait_frames - count)

        if not args.debug_mode:
            cv2.imshow("Detection Results", frame)

        if count <= args.wait_frames:
            print("waiting")
        else:
            note_change = music_obj.add_pose(points_return, count - args.wait_frames)
            if args.debug_mode and note_change is not None:
                add_note_change(frame, note_change)

        if args.debug_mode:
            cv2.imshow("Detection Results", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


def normalize(x, y, w, h):

    x_new = -1 + 2*(float(x)/float(w))
    y_new = -1 + 2*(float(y)/float(h))
    return [x_new, y_new]

def add_keypoints(frame, x, y, i):
    cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


def add_note_change(frame, note_change):
    if note_change:
        note_message = "NOTE CHANGED"
        cv2.putText(frame, note_message, (15, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)
    else:
        note_message = "NOTE DIDN'T CHANGE"
        cv2.putText(frame, note_message, (15, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)


def draw_skeleton(POSE_PAIRS, frame, points, prev_frame_time, idx):
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    # Draw performance stats
    # inf_time_message = "Inference time: N\A for async mode"
    # render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
    # async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id)
    fps_message = "FPS {}".format(fps)
    if idx >= 0 :
        start_messgae = "Going to start in {} frames".format(idx)
        cv2.putText(frame, start_messgae, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    # cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    # cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    # cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
    #             (10, 10, 200), 1)
    cv2.putText(frame, fps_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    return prev_frame_time

