"""
calibrate_intrinsics.py and run_repeated_calibrations.sh produce a series of
pickle files in directories where images are found (a.k.a. very scattered).
The filenames will have the pattern
    f"randomrun_{len(sampled_paths)}samples_{timestamp()}.pickle"
"""
import argparse
import pickle
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        prog="visualize_repeated_calibrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datadir",
        type=Path,
        help="Root directory where all images are stored downstream of",
    )
    args = parser.parse_args()
    return args


def project_pts(extrinsics, square_size, num_sqares_x, num_squares_y):
    """
    extrinsics : np.ndarray
        The (3,4) matrix for one set of points
    square_size : float
        The size of the square in meters
    num_squares_x :
        The number of checkerboard squares in x
    num_squares_x :
        The number of checkerboard squares in y
    """
    # Create a bunch of points in the z=0 plane
    # Project them based on extrinsics
    width = board_width * square_size
    height = board_height * square_size

    # draw calibration board
    X_board = np.ones((4, 5))
    # X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3, 0] = [0, 0, 0]
    X_board[0:3, 1] = [width, 0, 0]
    X_board[0:3, 2] = [width, height, 0]
    X_board[0:3, 3] = [0, height, 0]
    X_board[0:3, 4] = [0, 0, 0]


if __name__ == "__main__":
    args = parse_args()
    all_pickles = glob(
        str(args.datadir.joinpath("**/randomrun*pickle")), recursive=True
    )
    for p in all_pickles:
        with open(p, "rb") as infile:
            data = pickle.load(infile)
            objpoints = data["objpoints"]
            rvecs = data["rvecs"]
            tvecs = data["tvecs"]
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            for i in range(len(rvecs)):
                R, _ = cv2.Rodrigues(rvecs[i])
                M = np.eye(4)
                M[:3, :3] = R
                t = tvecs[i]
                M[:3, 3:4] = t
                M = M[:3]
                objpoints = data["objpoints"][0]
                objpoints_homog = np.concatenate(
                    (objpoints, np.ones((objpoints.shape[0], 1))), axis=1
                )
                projected = M @ objpoints_homog.T

                ax.scatter(projected[0], projected[1], projected[2])

                ax.set_xlabel("X Label")
                ax.set_ylabel("Y Label")
                ax.set_zlabel("Z Label")

            plt.show()
