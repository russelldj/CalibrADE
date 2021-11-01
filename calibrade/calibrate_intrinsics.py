import glob
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from constants import DATA_FOLDER

# termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Visualization interval in seconds
PAUSE_INTERVAL = 2
logger = logging.getLogger(__name__)

# Taken from
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
def calibrate_images(image_names, num_grid_corners=(7, 6), vis=True):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_grid_corners[0] * num_grid_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : num_grid_corners[0], 0 : num_grid_corners[1]].T.reshape(
        -1, 2
    )

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in image_names:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, num_grid_corners, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # TODO determine these constants
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, num_grid_corners, corners2, ret)
            title = "Detected control points"
        else:
            title = "Failed control point detection"
        if vis:
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
    }
    return calibration_params


def visualize_undistortion(image_path, mtx, dist):
    # Undistort
    img = cv2.imread(image_path)
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


def calculate_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    average_error = total_error / len(objpoints)
    return average_error


if __name__ == "__main__":
    calibration_image_paths = glob.glob(
        os.path.join(DATA_FOLDER, "opencv_examples", "left*.jpg")
    )
    calibration_params = calibrate_images(calibration_image_paths)

    undistortion_image_path = os.path.join(DATA_FOLDER, "opencv_examples", "left12.jpg")
    visualize_undistortion(
        undistortion_image_path, calibration_params["mtx"], calibration_params["dist"]
    )
    average_error = calculate_error(**calibration_params)
    print(f"Average error is {average_error} pixels")
