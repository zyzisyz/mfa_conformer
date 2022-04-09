#!/usr/bin/env python
# encoding: utf-8

import argparse
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str,
                        default="datasets/VoxCeleb/voxceleb1/")
    parser.add_argument('--src_trials_path', help='src_trials_path',
                        type=str, default="voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trials_path', help='dst_trials_path',
                        type=str, default="data/trial.lst")
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)

    f = open(args.dst_trials_path, "w")
    for item in trials:
        enroll_path = os.path.join(
            args.voxceleb1_root, "wav", item[1])
        test_path = os.path.join(args.voxceleb1_root, "wav", item[2])
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))
