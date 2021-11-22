import argparse
import cv2
from pathlib import Path

from root import read_data
from calibrate_intrinsics import NUM_GRID_CORNERS
from util import get_cached_corners


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=Path, help="directory containing all images")
    args = parser.parse_args()
    image_paths = read_data(args.datadir)
    length = len(image_paths)
    for i, image_name in enumerate(image_paths):
        if i % 10 == 0:
            print(f"Finding corners in {i} / {length}")
        gray = cv2.cvtColor(cv2.imread(str(image_name)), cv2.COLOR_BGR2GRAY)
        get_cached_corners(image_name, gray, NUM_GRID_CORNERS)
