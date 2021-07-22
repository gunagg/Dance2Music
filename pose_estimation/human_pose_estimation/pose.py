import sys
import os
from argparse import ArgumentParser

import cv2
import time
import logging as log
import numpy as np
import simpleaudio
from openvino.inference_engine import IENetwork, IEPlugin
import sys


def get_pose(args):
    if args.nn:
        # from music_main_nn import Music
        from music_main_nn_over_BS import Music
    else:
        from music_main import Music


    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    render_time = 0
    ret, frame = cap.read()
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    count = 0
    if args.nn:
        music_obj = Music(args.nn_frame_start, args.music_freq, args.joint_idx)
    else:
        music_obj = Music(args.pca_frame_freq, args.music_freq, args.joint_idx, args.no_pca)
    pca_joints = [int(val) for val in args.pca_joints.split(",")]
    pca_exclude_joints = [int(val) for val in args.pca_exclude_joints.split(",")]
    debug_mode = args.debug_mode

    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break
        count += 1

        inf_start = time.time()
        # in_frame = cv2.resize(next_frame, (w, h))
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs
            pw_relations = res['Mconv7_stage2_L1']
            kp_heatmaps = res['Mconv7_stage2_L2']
            """
            Nose 0, Neck 1, Right Shoulder 2, Right Elbow 3, Right Wrist 4,
            Left Shoulder 5, Left Elbow 6, Left Wrist 7, Right Hip 8,
            Right Knee 9, Right Ankle 10, Left Hip 11, Left Knee 12,
            LAnkle 13, Right Eye 14, Left Eye 15, Right Ear 16,
            Left Ear 17, Background 18
            """
            nPoints = 18
            POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                          [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                          [0, 14], [0, 15], [14, 16], [15, 17]]
            threshold = 0.2

            points = []
            points_return = []
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = kp_heatmaps[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                x = frame.shape[1] / probMap.shape[1] * point[0]
                y = frame.shape[0] / probMap.shape[0] * point[1]

                if prob > threshold:
                    # if True:  # Toggle circles and labels
                    if i == 0:
                        probMap_16 = kp_heatmaps[0, 16, :, :]
                        _, _, _, point_16 = cv2.minMaxLoc(probMap_16)
                        x_16 = frame.shape[1] / probMap_16.shape[1] * point_16[0]
                        y_16 = frame.shape[0] / probMap_16.shape[0] * point_16[1]

                        probMap_17 = kp_heatmaps[0, 17, :, :]
                        _, _, _, point_17 = cv2.minMaxLoc(probMap_17)
                        x_17 = frame.shape[1] / probMap_17.shape[1] * point_17[0]
                        y_17 = frame.shape[0] / probMap_17.shape[0] * point_17[1]

                        rect_size = int(abs(x_17 - x_16)/2.0) + 5
                        frame[int(y)-rect_size:int(y)+rect_size,int(x)-rect_size:int(x)+rect_size,:] = 255.0

                    add_keypoints(frame, x, y, i)

                    # Add the point to the list if the probability is greater than the threshold

                    points.append((int(x), int(y)))
                    if args.nn:
                        points_return += normalize(int(x), int(y), frame.shape[1], frame.shape[0])

                    else:
                        if args.no_pca:
                            points_return += normalize(int(x), int(y), frame.shape[1], frame.shape[0])
                        else:
                            if i in pca_exclude_joints:
                                points_return += [-1, -1]
                            else:
                                if pca_joints == "-1" or i in pca_joints:
                                    points_return += normalize(int(x), int(y), frame.shape[1], frame.shape[0])
                                else:
                                    points_return += [-1, -1]
                    # if i == 4:
                    #     print(x, y, points_return[-2], points_return[-1])
                else:
                    points.append(None)
                    points_return += [-1, -1]

            # Draw Skeleton
            prev_frame_time = draw_skeleton(POSE_PAIRS, frame, points, prev_frame_time, args.wait_frames - count)
            # combine_PCA_audio.main(points)

        # Resize output frame
        # frame = cv2.resize(frame, (1920, 1080))

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

        cur_request_id, next_request_id = next_request_id, cur_request_id
        frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    del exec_net
    del plugin


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
