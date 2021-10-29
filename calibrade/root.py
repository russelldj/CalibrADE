from ast import parse
import numpy as np
import argparse
from pathlib import Path
from prune import *

# read images names from data folder
def read_data(datadir):
    # read all image names and create path vector
    img_paths = []
    extensions = {"*.jpg", "*.JPG", "*png", "*.PNG"}
    for ext in extensions:
        img_paths.extend(list(datadir.glob(ext)))

    return img_paths


# pruning
def prune_wrapper(img_paths):
    prune_ids_init = np.zeros(len(img_paths), dtype=bool)
    prune_ids = prune(img_paths)

    return prune_ids


# deciding the subset test split


def select_subset(split_perc, ids):

    subset_ids = np.zeros(len(ids), dtype=bool)
    subset_size = np.floor(split_perc * np.sum(ids)) / 100
    options = np.where(ids)[0]
    subset_ids[np.random.choice(options, size=subset_size, replace=False)] = True

    return subset_ids


# additional visualize dataset dump (common for test or train set)
def visualize_set(img_paths, ids, tag):  # tag decides if its train or test folder
    # optional
    print("tbd")


def optimize(img_paths, test_ids, global_train_ids, max_iter, train_subset_perc):

    test_err = []
    cam_params = []
    train_subset_ids_list = []
    for _ in range(0, max_iter):

        # random subset of train ids
        train_ids = select_subset(train_subset_perc, global_train_ids)
        train_subset_ids_list.append(train_ids)
        # calibration
        cam_params.append(calibrate(img_paths, train_ids))
        # evaluation
        test_err.append(evaluate(img_paths, test_ids, cam_params[-1]))

    iter = np.argmin(test_err)
    return cam_params[iter], train_subset_ids_list[iter]


def root(args):
    img_paths = read_data(args.datadir)
    prune_ids = prune_wrapper(img_paths)
    test_ids = select_subset(args.test_set_split_perc, prune_ids)
    global_train_ids = np.invert(test_ids)
    cam_params, _ = optimize(
        img_paths, test_ids, global_train_ids, args.max_iter, args.train_subset_perc
    )


def args_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", default="./data/", help="directory containing all images"
    )
    parser.add_argument(
        "--test_set_split_perc",
        default=50,
        help="what percentage of pruned set should be selected as initial test set",
    )
    parser.add_argument(
        "--dump_test_set",
        action="store_true",
        help="save test set images in a folder for visualization",
    )
    parser.add_argument(
        "--train_subset_perc",
        default=70,
        help="percentage of samples from global train set to be sampled for an optimization iteration",
    )
    parser.add_argument(
        "--max_iter", default=100, help="max number of optimization iterations"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    root(args_parse())
