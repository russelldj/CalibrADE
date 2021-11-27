import argparse
import pdb
from ast import parse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.optimize


from calibrate_intrinsics import calibrate, compute_extrinsics, evaluate_reprojection
from constants import SQUARE_SIZE
from prune import prune
from util import read_data
from visualize import visualize


# pruning
def prune_wrapper(img_paths, num_grid_corners):
    prune_ids_init = np.zeros(len(img_paths), dtype=bool)
    prune_ids = prune(img_paths, num_grid_corners)

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


def optimize_GA(
    img_paths,
    test_ids,
    global_train_ids,
    max_iter,
    train_subset_num,
    vis_hist=False,
    vis_extrinsics=False,
    num_grid_corners=(7, 9),
    square_size=SQUARE_SIZE,
    **kwargs,
):
    cached_images = {}
    test_errs = []

    # Only use training images for training
    train_img_paths = img_paths[global_train_ids]

    # Constrain all decision variables to be in the range (0, 1)
    bounds = [(0, 1)] * train_img_paths.shape[0]

    # Takes a vector of decision variables represenenting which camera to select and returns the
    # test set error. Note that this takes values from the local scope but these are not optimized over.
    def compute_fitness(x):
        sorted_inds = np.argsort(x)
        top_inds = sorted_inds[-train_subset_num:]
        train_ids = np.zeros(x.shape, dtype=bool)
        train_ids[top_inds] = True
        calibrated_params = calibrate(
            train_img_paths,
            train_ids,
            cached_images,
            num_grid_corners,
            square_size=square_size,
        )
        test_err = evaluate_reprojection(
            img_paths, test_ids, calibrated_params, cached_images
        )
        test_errs.append(test_err)
        return test_err

    # Run the genetic algorithm
    res = scipy.optimize.differential_evolution(
        compute_fitness, bounds=bounds, disp=True, maxiter=max_iter, popsize=1,
    )

    solution = res.x
    # Recompute the calibration solution since it is not directly reported
    sorted_inds = np.argsort(solution)
    top_inds = sorted_inds[-train_subset_num:]
    train_ids = np.zeros(solution.shape, dtype=bool)
    train_ids[top_inds] = True
    # Rerun calibration with chosen inds
    calibrated_params = calibrate(
        img_paths, train_ids, cached_images, num_grid_corners, square_size=square_size,
    )

    if vis_extrinsics:
        rvecs = calibrated_params["rvecs"]
        tvecs = calibrated_params["tvecs"]

        visualize(
            rvecs,
            tvecs,
            calibrated_params["mtx"],
            board_height=num_grid_corners[0],
            board_width=num_grid_corners[1],
            pattern_centric=False,
            square_size=square_size,
            image_shape=(1920, 1080),
        )

    if vis_hist:
        plt.hist(test_errs)
        plt.show()

    if "savepath" in kwargs and kwargs["savepath"] is not None:
        plt.scatter(np.arange(len(test_errs)), test_errs)
        plt.ylabel("Objective function value")
        plt.xlabel("Function evaluation")
        plt.savefig(kwargs["savepath"])

    return calibrated_params, train_ids


def optimize_random(
    img_paths,
    test_ids,
    global_train_ids,
    max_iter,
    train_subset_num,
    vis_hist=False,
    vis_extrinsics=False,
    num_grid_corners=(7, 9),
    square_size=SQUARE_SIZE,
    **kwargs,
):

    cached_images = {}
    test_err = []
    cam_params = []
    train_subset_ids_list = []
    for i in tqdm(range(0, max_iter), desc="Main optimization loop"):
        train_ids = select_subset_num(train_subset_num, global_train_ids)
        train_subset_ids_list.append(train_ids)
        cam_params.append(
            calibrate(
                img_paths,
                train_ids,
                cached_images,
                num_grid_corners,
                square_size=square_size,
            )
        )
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
                pattern_centric=False,
                square_size=square_size,
                image_shape=(1920, 1080),
            )

    if vis_hist:
        plt.hist(test_err)
        plt.show()

    min_idx = np.argmin(test_err)
    return cam_params[min_idx], train_subset_ids_list[min_idx]


OPTIMIZATION_TYPES = {"GA": optimize_GA, "random": optimize_random}


def root(args):
    print("Reading data... ", end="")
    img_paths = read_data(args.datadir)
    print(f"{len(img_paths)} images selected")
    assert len(img_paths) > 0, f"Found 0 images in {args.datadir}"

    print("Selecting a pruned set")
    prune_ids = prune_wrapper(img_paths, args.num_grid_corners)

    print("Selecting a subset from pruned as the test set")
    test_ids = select_subset_perc(args.test_set_split_perc, prune_ids)
    global_train_ids = np.invert(test_ids)

    print("Calling calibration on a series of images")
    optimization_func = OPTIMIZATION_TYPES[args.optimization_type]
    cam_params, _ = optimization_func(
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

    parser = argparse.ArgumentParser(
        prog="root", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("datadir", type=Path, help="directory containing all images")
    parser.add_argument(
        "-t",
        "--test-set-split-perc",
        default=50,
        type=float,
        help="what percentage of pruned set should be selected as initial test set",
    )
    parser.add_argument(
        "-o",
        "--optimization-type",
        default="random",
        choices=OPTIMIZATION_TYPES.keys(),
        help="Which type of optimization routine to use",
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
        default=10,
        type=int,
        help="number of samples from global train set to be sampled for an optimization iteration",
    )
    parser.add_argument(
        "-m",
        "--max-iter",
        default=10,
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
    parser.add_argument(
        "--savepath", help="Where to save the visualization result",
    )
    args = parser.parse_args()
    assert args.datadir.is_dir()

    return args


if __name__ == "__main__":
    root(args_parse())
