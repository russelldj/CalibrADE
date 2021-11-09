import argparse
import pdb
from ast import parse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from calibrate_intrinsics import calibrate, compute_extrinsics, evaluate_reprojection
from opencv_visualize_extrinsics import visualize
from prune import prune
from constants import SQUARE_SIZE


# read images names from data folder
def read_data(datadir):
    # read all image names and create path vector
    img_paths = []
    extensions = {"*.jpg", "*.JPG", "*png", "*.PNG"}
    for ext in extensions:
        img_paths.extend(list(datadir.glob(ext)))
    img_paths = np.asarray(img_paths)
    return img_paths


# pruning
def prune_wrapper(img_paths):
    prune_ids_init = np.zeros(len(img_paths), dtype=bool)
    prune_ids = prune(img_paths)

    return prune_ids


def select_subset_num(subset_size, ids):
    subset_ids = np.zeros(len(ids), dtype=bool)
    options = np.where(ids)[0]
    subset_ids[np.random.choice(options, size=subset_size, replace=False)] = True
    return subset_ids


# deciding the subset test split
def select_subset_perc(split_perc, ids):
    subset_size = int(split_perc / 100 * np.sum(ids))
    return select_subset_num(subset_size, ids)


# additional visualize dataset dump (common for test or train set)
def visualize_set(img_paths, ids, tag):  # tag decides if its train or test folder
    # optional
    print("tbd")


def optimize(
    img_paths,
    test_ids,
    global_train_ids,
    max_iter,
    train_subset_num,
    vis_hist=False,
    vis_extrinsics=False,
    num_grid_corners=(7, 9),
    square_size=SQUARE_SIZE,
):

    cached_images = {}
    test_err = []
    cam_params = []
    train_subset_ids_list = []
    for _ in range(0, max_iter):
        print(f"Running iteration {_} / {max_iter}")
        # random subset of train ids
        train_ids = select_subset_num(train_subset_num, global_train_ids)
        train_subset_ids_list.append(train_ids)
        # calibration
        print("Calibrating")
        cam_params.append(
            calibrate(
                img_paths,
                train_ids,
                cached_images,
                num_grid_corners,
                square_size=square_size,
            )
        )
        # evaluation
        print("Evaluating")
        test_err.append(
            evaluate_reprojection(img_paths, test_ids, cam_params[-1], cached_images)
        )
        if vis_extrinsics:
            rvecs = cam_params[-1]["rvecs"]
            tvecs = cam_params[-1]["tvecs"]

            visualize(
                rvecs,
                tvecs,
                cam_params[-1]["mtx"],
                board_height=num_grid_corners[0],
                board_width=num_grid_corners[1],
            )

    if vis_hist:
        plt.hist(test_err)
        plt.show()

    min_idx = np.argmin(test_err)
    return cam_params[min_idx], train_subset_ids_list[min_idx]


def root(args):
    print("Reading data... ", end="")
    img_paths = read_data(args.datadir)
    print(f"{len(img_paths)} images selected")
    assert len(img_paths) > 0, f"Found 0 images in {args.datadir}"

    print("Selecting a pruned set")
    prune_ids = prune_wrapper(img_paths)

    print("Selecting a subset from pruned as the test set")
    test_ids = select_subset_perc(args.test_set_split_perc, prune_ids)
    global_train_ids = np.invert(test_ids)

    print("Calling calibration on a series of images")
    cam_params, _ = optimize(
        img_paths=img_paths,
        test_ids=test_ids,
        global_train_ids=global_train_ids,
        max_iter=args.max_iter,
        train_subset_num=args.train_subset_num,
        vis_hist=args.vis_optimize_hist,
        vis_extrinsics=args.vis_extrinsics,
        num_grid_corners=args.num_grid_corners,
        square_size=args.square_size,
    )


def args_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=Path, help="directory containing all images")
    parser.add_argument(
        "-t",
        "--test-set-split-perc",
        default=50,
        type=float,
        help="what percentage of pruned set should be selected as initial test set",
    )
    parser.add_argument(
        "-d",
        "--dump-test-set",
        action="store_true",
        help="save test set images in a folder for visualization",
    )
    parser.add_argument(
        "-H",
        "--vis-optimize-hist",
        action="store_true",
        help="Plot the histogram visualization from optimization",
    )
    parser.add_argument(
        "-e",
        "--vis-extrinsics",
        action="store_true",
        help="Plot the histogram visualization from optimization",
    )
    parser.add_argument(
        "-r",
        "--train-subset-num",
        default=25,
        type=float,
        help="number of samples from global train set to be sampled for an optimization iteration",
    )
    parser.add_argument(
        "-m",
        "--max-iter",
        default=100,
        type=int,
        help="max number of optimization iterations",
    )
    parser.add_argument(
        "-c",
        "--num-grid-corners",
        default=(7, 9),
        type=int,
        nargs=2,
        help="The number of rows and columns of corners",
    )
    parser.add_argument(
        "-s",
        "--square-size",
        default=SQUARE_SIZE,
        type=float,
        help="The size of a single calibration square in meters",
    )
    args = parser.parse_args()
    assert args.datadir.is_dir()

    return args


if __name__ == "__main__":
    root(args_parse())
