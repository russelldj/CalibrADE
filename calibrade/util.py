import cv2
import numpy as np
import pickle
import time


"""
The purpose of this file is to make basic tools that can be easily imported.
This should not import any of our files.
"""


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


def read_data(datadir):
    """Read images names from data folder."""
    # read all image names and create path vector
    img_paths = []
    extensions = {"*.jpg", "*.JPG", "*png", "*.PNG"}
    for ext in extensions:
        img_paths.extend(list(datadir.glob(ext)))
    img_paths = np.asarray(img_paths)
    return img_paths


def timestamp():
    return int(time.time() * 1e6)
