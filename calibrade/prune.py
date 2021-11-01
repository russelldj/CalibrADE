import argparse
from pathlib import Path
import cv2
from matplotlib import pyplot
import numpy
from skimage.morphology import binary_closing, binary_opening


# The LAP2 filter measurement from here:
# https://www.sciencedirect.com/science/article/pii/S0031320312004736
# The modified Laplacian if an image is computed as
#   | I * Lx | + | I * Ly |
# Where I is the image, * is convolution, Lx is [-1, 2, -1], and Ly is Lx.T
KERNEL = numpy.array([-1, 2, -1])


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


# TODO: Make this real
def target_mask(image):
    """
    Arguments:
        image: grayscale image of shape (N, M)

    Returns: Boolean mask of shape (N, M), which is True for pixels that
        we think are on the calibration target.
    """
    return numpy.ones(image.shape[:2], dtype=bool)


def prune(image_paths, ratio=0.5, plot=False, save_images=False):
    """
    Arguments:
        image_paths: list of strings of paths to the images
        ratio: (float) fraction of the images that we are asserting to be of
            good blur quality
        plot: (bool) show a plot of the blur values across the images (debug)
        save_images: (bool) NOT IMPLEMENTED CURRENTLY - save visualization
            images to double-check this process. SEE COMMENTED OUT CODE FOR
            A PARTIAL IMPLEMENTATION.
    """

    # Calculate the "focus" metric for each image
    focus = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = target_mask(image)
        focus.append(lap2_focus_measure(image.astype(float), mask))
    focus = numpy.array(focus)

    # TODO: Make this sort of result visualization relevant again
    # # Temporary code to evaluate images. Later, will turn into good/bad sorting
    # # When only sorting images, perhaps use a copy tool?
    # if save_images:
    #     for path, value in zip(image_paths, focus):
    #         image = cv2.imread(path)
    #         image = cv2.putText(
    #             image,
    #             f"{value:.3E}",
    #             (10, 100),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=2,
    #             color=(0, 255, 0),
    #             thickness=2,
    #         )

    #         grey_mask = target_mask(image).astype(numpy.uint8) * 255
    #         color_mask = cv2.cvtColor(grey_mask, cv2.COLOR_GRAY2BGR)
    #         image = numpy.hstack((image, color_mask))

    #         newpath = good_dir.joinpath(Path(path).name)
    #         cv2.imwrite(str(newpath), image)
    #         print(f"Saved {newpath}")

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
        "-s", "--save", help="Save images to the folders", action="store_true"
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
