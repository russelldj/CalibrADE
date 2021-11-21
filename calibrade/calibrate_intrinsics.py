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
from util import read_data, timestamp


# termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Visualization interval in seconds
PAUSE_INTERVAL = 2
logger = logging.getLogger(__name__)

NUM_GRID_CORNERS = (7, 9)


def get_cached_corners(image_path, gray, num_grid_corners):

    pickle_path = image_path.with_name("chessboard_corners.pickle")
    try:
        with open(pickle_path, "rb") as handle:
            saved = pickle.load(handle)
    except FileNotFoundError:
        saved = dict()
        with open(pickle_path, "wb") as handle:
            pickle.dump(saved, handle)

    try:
        return saved[image_path.name]
    except KeyError:
        ret, corners = cv2.findChessboardCorners(gray, num_grid_corners, None)
        # Do subpixel refinement
        if ret:
            # TODO determine these constants
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        saved[image_path.name] = (ret, corners)
        with open(pickle_path, "wb") as handle:
            pickle.dump(saved, handle)
        return (ret, corners)


def read_cached_image(fname, cached_images):
    try:
        return cached_images[str(fname)]
    except KeyError:
        image = cv2.imread(str(fname))
        cached_images[str(fname)] = image
        return image


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
    return calibration_params


def calibrate(image_names, train_ids, cached_images, num_grid_corners, **kwargs):
    """
    Thin wrapper around calibrate_images which selects from the valid set
    """
    valid_images = image_names[train_ids]
    calib_results = calibrate_images(
        valid_images, cached_images, num_grid_corners=num_grid_corners, **kwargs
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


def calculate_reprojection_error(
    objpoints, imgpoints, rvecs, tvecs, mtx, dist, **kwargs
):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = np.linalg.norm(imgpoints[i] - imgpoints2) / len(imgpoints2)
        total_error += error
    try:
        average_error = total_error / len(objpoints)
    except ZeroDivisionError:
        average_error = np.inf
    return average_error


# def evaluate_reprojection(image_paths, test_ids, params, cached_images):
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


def evaluate_reprojection(image_paths, test_ids, params, cached_images):

    mtx = params["mtx"]
    dist = params["dist"]

    rvecs, tvecs, successful_objpoints, successful_imgpoints = compute_extrinsics(
        image_paths, test_ids, params, cached_images
    )
    # Consider renaming mtx and dist
    # TODO, also report the number that were correctly triangulated
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

    # Take a sampled set of images if asked for them
    image_paths = read_data(args.datadir)

    if args.sample_number > 0:
        sampled_paths = np.random.choice(
            image_paths, size=args.sample_number, replace=False
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
    save_path = args.datadir.joinpath(
        f"randomrun_{len(sampled_paths)}samples_{timestamp()}.pickle"
    )
    with open(save_path, "wb") as handle:
        pickle.dump(save_data, handle)
