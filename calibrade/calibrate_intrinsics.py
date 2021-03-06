import argparse
import glob
import logging
import os
import pdb
import pickle
from pathlib import Path
from sys import exit
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import DATA_FOLDER, SQUARE_SIZE
from prune import prune
from util import get_cached_corners, read_data, timestamp


# termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Visualization interval in seconds
PAUSE_INTERVAL = 2
CORNER_POINT_VIS_SIZE = 10
logger = logging.getLogger(__name__)

NUM_GRID_CORNERS = (7, 9)


def read_cached_image(fname, cached_images):
    try:
        return cached_images[str(fname)]
    except KeyError:
        image = cv2.imread(str(fname))
        cached_images[str(fname)] = image
        return image


def sharpness_downsample(image_paths, ratio):
    good_mask = prune(image_paths, NUM_GRID_CORNERS, ratio=ratio)
    # Check that we hit our desired ratio to within 2% (chosen arbitrarily)
    end_fraction = 1 - (np.sum(good_mask) / len(good_mask))
    assert np.isclose(end_fraction, ratio, atol=0.02), (
        f"Sharpness downsampling was supposed to leave {ratio} fraction of"
        + f" the images, but it actually left {end_fraction}"
    )
    return image_paths[good_mask]


# Taken from
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
def calibrate_images(
    image_names,
    cached_images,
    num_grid_corners=NUM_GRID_CORNERS,
    square_size=SQUARE_SIZE,
    vis=False,
):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_grid_corners[0] * num_grid_corners[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : num_grid_corners[0], 0 : num_grid_corners[1]].T.reshape(-1, 2)
        * square_size
    )

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in image_names:
        img = read_cached_image(fname, cached_images)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = get_cached_corners(fname, gray, num_grid_corners)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        if vis:
            if ret:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, num_grid_corners, corners, ret)
                title = "Detected control points"
            else:
                title = "Failed control point detection"
            img = np.flip(img, 2)  # convert from BGR to RGB
            plt.imshow(img)
            plt.title(title)
            plt.pause(PAUSE_INTERVAL)

    # Calibration
    # TODO determine the convention for the rotation vectors
    if len(imgpoints) > 1:
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        calibration_params = {
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "objpoints": objpoints,
            "imgpoints": imgpoints,
            "num_grid_corners": num_grid_corners,
            "square_size": square_size,
        }
    else:
        calibration_params = {
            "objpoints": objpoints,
            "imgpoints": imgpoints,
            "num_grid_corners": num_grid_corners,
            "square_size": square_size,
        }
        print("Not enough images")

    return calibration_params


def calibrate(image_names, train_ids, cached_images, num_grid_corners, **kwargs):
    """
    Thin wrapper around calibrate_images which selects from the valid set
    """
    valid_images = image_names[train_ids]
    calib_results = calibrate_images(
        valid_images, cached_images, num_grid_corners=num_grid_corners, **kwargs,
    )
    return calib_results


def visualize_undistortion(image_path, mtx, dist, vis_path=None):
    # Undistort
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort method 1
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    if vis_path is None:
        plt.imshow(np.flip(dst, axis=2))
        plt.title("Undistorted image")
        plt.pause(PAUSE_INTERVAL)
    else:
        dst = cv2.resize(dst, dsize=img.shape[:2][::-1])
        cv2.imwrite(str(vis_path), np.hstack((img, dst)))


def project_points(objpoints, rvecs, tvecs, mtx, dist):
    projected_pts = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        projected_pts.append(projected)
    return projected_pts


def calculate_reprojection_error(
    objpoints, imgpoints, rvecs, tvecs, mtx, dist, **kwargs
):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        errors = np.linalg.norm(np.squeeze(imgpoints[i] - imgpoints2), axis=1)
        error = np.mean(errors)
        if "image_paths" in kwargs:
            image_file = kwargs["image_paths"][kwargs["test_ids"]][i]
            img = read_cached_image(image_file, kwargs["cached_images"])
            plt.scatter(
                imgpoints[i][..., 0],
                imgpoints[i][..., 1],
                c="b",
                s=CORNER_POINT_VIS_SIZE * 30,
                label="Detected",
            )
            plt.scatter(
                imgpoints2[..., 0],
                imgpoints2[..., 1],
                c="r",
                s=CORNER_POINT_VIS_SIZE * 30,
                label="Projected",
            )
            # plt.scatter(
            #    imgpoints[i][..., 0],
            #    imgpoints[i][..., 1],
            #    s=CORNER_POINT_VIS_SIZE * 4,
            #    linewidths=1,
            #    c="g",
            #    marker="X",
            #    label="Detected",
            # )
            # plt.plot(
            #    imgpoints2[..., 0],
            #    imgpoints2[..., 1],
            #    markersize=CORNER_POINT_VIS_SIZE,
            #    linestyle="none",
            #    markeredgewidth=1.5,
            #    marker="o",
            #    markerfacecolor="none",
            #    markeredgecolor="fuchsia",
            #    label="Projected",
            # )
            plt.legend(prop={"size": 15})
            plt.imshow(img)
            plt.show()
        total_error += error

    try:
        average_error = total_error / len(objpoints)
    except ZeroDivisionError:
        average_error = np.inf
    return average_error


def compute_extrinsics(image_paths, test_ids, params, cached_images):
    valid_images = image_paths[test_ids]

    objpoints = params["objpoints"]
    num_grid_corners = params["num_grid_corners"]
    mtx = params["mtx"]
    dist = params["dist"]

    # The detected corners in the image
    all_imgpoints = []

    for fname in valid_images:
        img = read_cached_image(fname, cached_images)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = get_cached_corners(fname, gray, num_grid_corners)
        if ret:
            all_imgpoints.append(corners)

    # The object is the same in all frames. Replicate it for each image
    all_objpoints = [objpoints[0]] * len(all_imgpoints)
    rvecs = []
    tvecs = []

    successful_objpoints = []
    successful_imgpoints = []
    for objpoints, imgpoints in zip(all_objpoints, all_imgpoints):
        ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
            successful_objpoints.append(objpoints)
            successful_imgpoints.append(imgpoints)
    return rvecs, tvecs, successful_objpoints, successful_imgpoints


def evaluate_reprojection(
    image_paths, test_ids, params, cached_images, vis_images=False
):
    try:
        mtx = params["mtx"]
        dist = params["dist"]
    except KeyError:
        # This means that the relevant parameters weren't computed
        return np.inf

    rvecs, tvecs, successful_objpoints, successful_imgpoints = compute_extrinsics(
        image_paths, test_ids, params, cached_images
    )
    # Consider renaming mtx and dist
    # TODO, also report the number that were correctly triangulated
    if vis_images:
        error = calculate_reprojection_error(
            successful_objpoints,
            successful_imgpoints,
            rvecs,
            tvecs,
            mtx,
            dist,
            image_paths=image_paths,
            cached_images=cached_images,
            test_ids=test_ids,
        )
    else:
        error = calculate_reprojection_error(
            successful_objpoints, successful_imgpoints, rvecs, tvecs, mtx, dist
        )

    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="calibrate_intrinsics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("datadir", type=Path, help="directory containing all images")
    parser.add_argument(
        "-p",
        "--sharpness-ratio",
        type=float,
        default=-1,
        help="Fraction (0-1) of the images that we are asserting to be of bad"
        " blur quality. For example, 0.25 means we will drop the bottom"
        " 25%% and sample from the top 75%%",
    )
    parser.add_argument(
        "-r",
        "--suppress-reprojection",
        action="store_true",
        help="Whether to suppress the reprojection printout (for repeated runs)",
    )
    parser.add_argument(
        "-s",
        "--sample-number",
        type=int,
        default=0,
        help="How many (randomly sampled) images to run calibration on",
    )
    parser.add_argument(
        "-v",
        "--visualize-undistortion",
        action="store_true",
        help="Whether to visualize undistorted images in /tmp/ (will print path)",
    )
    args = parser.parse_args()

    # Find all images in the given directory
    image_paths = read_data(args.datadir)

    # Downsample to a certain sharpness ratio if requested
    if args.sharpness_ratio > 0:
        image_paths = sharpness_downsample(image_paths, args.sharpness_ratio)
        if len(image_paths) < args.sample_number:
            print(
                f"WARNING! Ratio {args.sharpness_ratio} downsampled down to"
                f" {len(image_paths)} when we need to sample"
                f" {args.sample_number}, ending"
            )
            exit(0)

    # Take a sampled set of images if asked for them
    if args.sample_number > 0:
        sampled_paths = np.random.choice(
            image_paths, size=args.sample_number, replace=False
        )
        assert len(sampled_paths) == args.sample_number, (
            f"Tried to sample to {args.sampled_number}, but could only get"
            + f" {len(sampled_paths)} images with given settings"
        )
    else:
        sampled_paths = image_paths

    try:
        calibration_params = calibrate_images(sampled_paths, cached_images={})
    except cv2.error:
        print("!" * 80)
        print(f"WARNING! A calibration failed in {args.datadir}!")
        print(f"Images: {sampled_paths}")
        print("!" * 80)
        print(traceback.format_exc())
        print("!" * 80)
        exit(0)

    average_error = calculate_reprojection_error(**calibration_params)
    # TODO: look into whether average error is actually pixels
    if not args.suppress_reprojection:
        print(f"Average reprojection error is {average_error}px")

    # Choose a random set of images to visualize
    if args.visualize_undistortion:
        for image_choice in np.random.choice(image_paths, size=4, replace=False):
            nameized = str(image_choice).replace("/", "_")
            vis_path = Path(f"/tmp/{timestamp()}_{nameized}")
            visualize_undistortion(
                image_choice,
                calibration_params["mtx"],
                calibration_params["dist"],
                vis_path=vis_path,
            )
            print(f"Saved debug undistorted image {vis_path}")

    # Save certain data from this run for the future. Copy the variable for
    # name clarity
    save_data = calibration_params
    save_data["sampled_paths"] = sampled_paths
    save_data["average_error"] = average_error
    save_data["sharpness_ratio"] = args.sharpness_ratio
    save_path = args.datadir.joinpath(
        f"randomrun_{len(sampled_paths)}samples_{timestamp()}.pickle"
    )
    with open(save_path, "wb") as handle:
        pickle.dump(save_data, handle)
