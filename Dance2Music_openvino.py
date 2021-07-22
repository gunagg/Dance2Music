import sys
import os
from argparse import ArgumentParser

import cv2
import time
import logging as log
import numpy as np
import simpleaudio
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",default="pose_estimation/human_pose_estimation/models/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml",
                        help="Path to an .xml file with a trained model.", required=False, type=str)
    parser.add_argument("-i", "--input", default = "cam",
                        help="Path to video file or image. 'cam' for capturing video stream from internal camera.",
                        required=False, type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.",
                        type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir",
                        help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. "
                             "Demo will look for a suitable plugin for device specified (CPU by default)",
                        default="CPU", type=str)
    parser.add_argument("--wait_frames",
                        help="Number of frames to wait before starting",
                        default=100, type=int)
    parser.add_argument("--pca_frame_freq",
                        help="After how many frames to do pca training",
                        default=50, type=int)
    parser.add_argument("--nn_frame_start",
                        help="After how many frames to start nn",
                        default=36, type=int)
    parser.add_argument("--music_freq",
                        help="After how many frames to play music",
                        default=6, type=int)
    parser.add_argument("--joint_idx",
                        help="Which joint to use to play music initially",
                        default=9, type=int)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--nn', action='store_true', default=True)

    parser.add_argument('--debug_mode', action='store_true', default=False)

    parser.add_argument("--pca_joints",
                        help="Which joint to use to use for PCA, comma seperated, eg 1,4,7. -1 stands for all",
                        default="-1", type=str)

    parser.add_argument("--pca_exclude_joints",
                        help="Which joint to not use to use for PCA, comma seperated, eg 1,4,7. -1 stands for None",
                        default="-1", type=str)


    return parser


def main():
    args = build_argparser().parse_args()
    args.no_pca = not args.pca

    from pose_estimation.human_pose_estimation.pose import get_pose

    get_pose(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
