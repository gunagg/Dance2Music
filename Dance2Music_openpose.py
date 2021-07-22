import sys
import os
from argparse import ArgumentParser

import cv2
import time
import logging as log
import numpy as np
import simpleaudio


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default = "cam",
                        help="Path to video file or image. 'cam' for capturing video stream from internal camera.",
                        required=False, type=str)
    parser.add_argument("--nn_frame_start",
                        help="After how many frames to start nn",
                        default=36, type=int)
    parser.add_argument("--music_freq",
                        help="After how many frames to play music",
                        default=6, type=int)
    parser.add_argument("--wait_frames",
                        help="Number of frames to wait before starting",
                        default=10, type=int)
    parser.add_argument('--debug_mode', action='store_true', default=False)

    return parser


def main():
    args = build_argparser().parse_args()
    
    from pose_estimation.human_pose_estimation.pose_openpose import get_pose

    get_pose(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
