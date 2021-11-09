import glob
import logging
from pathlib import Path
import os
import pdb
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from constants import DATA_FOLDER

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
    image_names, cached_images, num_grid_corners=NUM_GRID_CORNERS, vis=False
):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_grid_corners[0] * num_grid_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : num_grid_corners[0], 0 : num_grid_corners[1]].T.reshape(
        -1, 2
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
    }
    return calibration_params


def calibrate(image_names, train_ids, cached_images, num_grid_corners):
    """
    Thin wrapper around calibrate_images which selects from the valid set
    """
    valid_images = image_names[train_ids]
    calib_results = calibrate_images(
        valid_images, cached_images, num_grid_corners=num_grid_corners
    )
    return calib_results


def visualize_undistortion(image_path, mtx, dist):
    # Undistort
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort method 1
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    plt.imshow(np.flip(dst, axis=2))
    plt.title("Undistorted image")
    plt.pause(PAUSE_INTERVAL)


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
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
    calibration_image_paths = glob.glob(
        os.path.join(DATA_FOLDER, "opencv_examples", "left*.jpg")
    )
    calibration_params = calibrate_images(calibration_image_paths)

    undistortion_image_path = os.path.join(DATA_FOLDER, "opencv_examples", "left12.jpg")
    visualize_undistortion(
        undistortion_image_path, calibration_params["mtx"], calibration_params["dist"]
    )
    average_error = calculate_reprojection_error(**calibration_params)
    print(f"Average error is {average_error} pixels")
