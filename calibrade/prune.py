import argparse
from pathlib import Path
import cv2
from matplotlib import pyplot
import numpy
import pickle
from scipy.spatial import ConvexHull
from skimage.morphology import binary_closing, binary_opening

from util import get_cached_corners


# The LAP2 filter measurement from here:
# https://www.sciencedirect.com/science/article/pii/S0031320312004736
# The modified Laplacian if an image is computed as
#   | I * Lx | + | I * Ly |
# Where I is the image, * is convolution, Lx is [-1, 2, -1], and Ly is Lx.T
KERNEL = numpy.array([-1, 2, -1])


def get_cached_focus(path, num_grid_corners):
    pickle_path = path.with_name("focus_metric.pickle")
    try:
        with open(pickle_path, "rb") as handle:
            saved = pickle.load(handle)
    except FileNotFoundError:
        saved = dict()
        with open(pickle_path, "wb") as handle:
            pickle.dump(saved, handle)

    try:
        return saved[path.name]
    except KeyError:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = target_mask(image, path, num_grid_corners)
        focus = lap2_focus_measure(image.astype(float), mask)
        saved[path.name] = focus
        with open(pickle_path, "wb") as handle:
            pickle.dump(saved, handle)
        return focus


def lap2_focus_measure(image, mask):
    """
    Arguments:
        image: (N, M) greyscale, floating point from 0-255
        mask: (N, M) boolean array, True where we want to extract values
    """
    lx = numpy.convolve(image.flatten(), KERNEL, mode="same").reshape(image.shape)
    ly = numpy.convolve(image.T.flatten(), KERNEL, mode="same").reshape(image.T.shape)
    raw = numpy.sum(numpy.abs(lx[mask]) + numpy.abs(ly[mask.T]))

    # vvvv Diverges from the paper vvvv

    # Because we are doing lap2 over a mask, we need to normalize by the size
    # of the mask! Otherwise large blurry things will have higher lap2 than
    # small sharp things
    number_norm = raw / numpy.sum(mask)
    # This feels kind of hand-wavy, but I noticed that dark images have a less
    # pronounced scores even if they look sharper to the human eye. When I
    # passed image/2 into the function, the score I got was 1/2 as much. Let's
    # scale by average brightness
    bright_norm = number_norm * (255 / numpy.average(image[mask]))
    # Remove nans
    no_nan = numpy.nan_to_num(bright_norm, nan=0.0)

    return no_nan


def cdf(x, normed=True, *args, **kwargs):
    x = sorted(x)
    y = numpy.arange(len(x))
    if normed:
        y /= len(x)
    return pyplot.plot(x, y, *args, **kwargs)


def target_mask(image, path, num_grid_corners):
    """
    Arguments:
        image: grayscale image of shape (N, M)
        path: pathlib.Path object for the image

    Returns: Boolean mask of shape (N, M), which is True for pixels that
        we think are on the calibration target.
    """
    ret, corners = get_cached_corners(
        image_path=path, gray=image, num_grid_corners=num_grid_corners
    )
    if ret:
        # Take the hull to get the outer 2D shape
        hull = ConvexHull(corners.squeeze())
        points2d = hull.points[hull.vertices]
        # Scale the points outward slightly
        scale = 1.3
        center = numpy.average(points2d, axis=0)
        for i in range(len(points2d)):
            points2d[i] = center + scale * (points2d[i] - center)
        # Clip to edges, note corners are (axis1, axis0)
        points2d[:, 0] = numpy.clip(points2d[:, 0], 0, image.shape[1] - 1)
        points2d[:, 1] = numpy.clip(points2d[:, 1], 0, image.shape[0] - 1)
        # Make a boolean mask
        mask = numpy.zeros(image.shape[:2], dtype=numpy.int32)
        # import ipdb; ipdb.set_trace()
        mask = cv2.fillPoly(
            mask, [points2d.reshape((-1, 1, 2)).astype(numpy.int32)], color=1.0
        )
        mask = mask.astype(bool)
    else:
        mask = numpy.ones(image.shape[:2], dtype=bool)
    return mask


def prune(image_paths, num_grid_corners, ratio=0.5, plot=False, save_image_dir=None):
    """
    Arguments:
        image_paths: list of strings of paths to the images
        num_grid_corners: int two-tuple like (7, 9) indicating the number of
            corners we want to find in the image. Used to try and make a mask
            of the board (area of interest)
        ratio: (float) fraction of the images that we are asserting to be of
            bad blur quality. 0.25 means we will drop the bottom 25% and
            take the top 75%.
        plot: (bool) show a plot of the blur values across the images (debug)
        save_image_dir: (None or pathlib.Path) If given a path, will saved
            images showing the target mask and focus value on them (slow)
    """

    # Calculate the "focus" metric for each image
    focus = numpy.array(
        [get_cached_focus(path, num_grid_corners) for path in image_paths]
    )

    # Code to evaluate images. Note particularly streamlined
    if save_image_dir:
        for path, value in zip(image_paths, focus):
            image = cv2.imread(str(path))
            image = cv2.putText(
                image,
                f"{value:.3E}",
                (10, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=2,
            )
            mask = target_mask(image, path, num_grid_corners)
            # import ipdb; ipdb.set_trace()
            masked_image = image[mask]
            masked_image[:, 1] = 255
            image[mask] = masked_image
            newpath = save_image_dir.joinpath(path.name)
            cv2.imwrite(str(newpath), image)
            print(f"Saved {newpath}")

    # Choose the images in the top ratio as the images we are marking as
    # good quality
    sorted_indices = numpy.argsort(focus)
    cutoff_idx = int(ratio * len(focus))

    if plot:
        cdf(focus, normed=False)
        threshold = focus[sorted_indices[cutoff_idx]]
        pyplot.plot([threshold] * 2, [0, len(focus)], "k--")
        pyplot.show()

    # Make a boolean array saying which ratio of the images are highest quality
    prune_ids = numpy.zeros(len(focus), dtype=bool)
    prune_ids[sorted_indices[cutoff_idx:]] = True
    return prune_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sorts given images ")
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Path to directory of images we want to search over",
    )
    parser.add_argument(
        "-f",
        "--filetype",
        help="Filetype for images, will be used in glob. For example: jpg/jpeg/png",
        default="jpg",
    )
    parser.add_argument(
        "-p", "--plot", help="Whether to plot results", action="store_true"
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=Path,
        help="Path to directory that you want to save debug images in (slow)",
    )
    args = parser.parse_args()
    assert args.image_dir.is_dir(), f"{args.image_dir} needs to be a directory"

    image_paths = [
        str(path.absolute())
        for path in sorted(list(args.image_dir.glob(f"*{args.filetype}")))
    ]
    assert (
        len(image_paths) > 0
    ), f"{str(args.image_dir.absolute())}/*{args.filetype} produced no files"

    print(prune(image_paths=image_paths, plot=args.plot, save_images=args.save))
