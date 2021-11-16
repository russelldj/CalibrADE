import numpy as np
import time


'''
The purpose of this file is to make basic tools that can be easily imported.
This should not import any of our files.
'''


# read images names from data folder
def read_data(datadir):
    # read all image names and create path vector
    img_paths = []
    extensions = {"*.jpg", "*.JPG", "*png", "*.PNG"}
    for ext in extensions:
        img_paths.extend(list(datadir.glob(ext)))
    img_paths = np.asarray(img_paths)
    return img_paths


def timestamp():
    return int(time.time() * 1e6)