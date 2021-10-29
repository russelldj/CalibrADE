import argparse
import cv2
from matplotlib import pyplot
import numpy
from pathlib import Path
import pickle
from skimage.morphology import binary_closing, binary_opening


# The LAP2 filter measurement from here:
# https://www.sciencedirect.com/science/article/pii/S0031320312004736
# The modified Laplacian if an image is computed as
#   | I * Lx | + | I * Ly |
# Where I is the image, * is convolution, Lx is [-1, 2, -1], and Ly is Lx.T
kernel = numpy.array([-1, 2, -1])
def lap2_focus_measure(image, mask):
    """
    Arguments:
        image: (N, M) greyscale, floating point from 0-255
        mask: (N, M) boolean array, True where we want to extract values
    """
    lx = numpy.convolve(image.flatten(), kernel, mode="same").reshape(image.shape)
    ly = numpy.convolve(image.T.flatten(), kernel, mode="same").reshape(image.T.shape)
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
    image needs to be a color image of shape (N, M, 3)

    Will return a boolean mask of shape (N, M), which is True for pixels that
    we think are panicle.
    """
    return numpy.ones(image.shape[:2], dtype=bool)


def get_threshold(image_keys, focus, ratio=0.2, step=0.05):
    """
    TODO.
    """
    frange = focus.max() - focus.min()
    thresh = frange * ratio + focus.min()

    highest_blur = -1
    blur_idx = -1
    lowest_sharp = 1e10
    sharp_idx = -1
    seen_blur = False
    seen_sharp = False

    # TODO: Is there a more ergonomic way to sort images? This section could
    # be polished. Can we climb up and down the CDF instead of the value?
    # E.g. indexing off of numpy.argsort.
    while not (seen_blur and seen_sharp):
        idx = numpy.argmin(numpy.abs(focus - thresh))
        image = cv2.imread(image_keys[idx])
        cv2.imshow("Does this count as blurry?", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        value = None
        while value not in ["y", "n"]:
            value = input(f"Threshold {thresh}: was it blurry? [y/n]:\n")
        if value == "y":
            if focus[idx] > highest_blur:
                highest_blur = focus[idx]
                blur_idx = idx
            seen_blur = True
            thresh += frange * step
        else:
            if focus[idx] < lowest_sharp:
                lowest_sharp = focus[idx]
                sharp_idx = idx
            seen_sharp = True
            thresh -= frange * step

    # TODO: Expand this to check a few more images around the critical point
    # to get more resolution
    return lowest_sharp


# TODO: Tune for time - downsample images? Even if only for mask generation?
def main(image_paths, good_dir, bad_dir, plot, recompute, save_images,
         threshold=None):

    # Convert to strings (keys for dictionary later)
    image_keys = [str(path.absolute()) for path in image_paths]

    # Save values for speed reasons on multiple runs
    pickle_path = Path('blur_values.pickle')
    if recompute or not pickle_path.is_file():
        metrics = {}
        for i, path in enumerate(image_keys):
            print(i)  # TODO: Remove or make more sophisticated
            image = cv2.imread(str(path))
            mask = target_mask(image)
            # TODO: Replace with a named tuple?
            metrics[path] = (
                lap2_focus_measure(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float),
                    mask,
                ),
                numpy.sum(mask),
            )
        with open(pickle_path, 'wb') as outfile:
            pickle.dump(metrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    # Make a hard line where we always work from these metrics
    with open(pickle_path, 'rb') as infile:
        metrics = pickle.load(infile)
    focus = numpy.array([metrics[path][0] for path in image_keys])
    # TODO: Once the pickle files with nans are all gone remove this call
    focus = numpy.nan_to_num(focus)
    num_panicle_pixels = numpy.array([metrics[path][1] for path in image_keys])

    # Temporary code to evaluate images. Later, will turn into good/bad sorting
    # When only sorting images, perhaps use a copy tool?
    if save_images:
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

            grey_mask = target_mask(image).astype(numpy.uint8) * 255
            color_mask = cv2.cvtColor(grey_mask, cv2.COLOR_GRAY2BGR)
            image = numpy.hstack((image, color_mask))

            newpath = good_dir.joinpath(path.name)
            cv2.imwrite(str(newpath), image)
            print(f"Saved {newpath}")

    # Make a decision about how much blur is too much
    if threshold is None:
        threshold = get_threshold(image_keys, focus)

    if plot:
        cdf(focus, normed=False)
        pyplot.plot([threshold]*2, [0, len(focus)], "k--")
        pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sorts given images "
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Path to directory of images we want to search over"
    )
    parser.add_argument(
        "good_dir",
        type=Path,
        help="Path to directory where we want to sort images deemed good. Can"
             " already exist, will overwrite images there if names match."
    )
    parser.add_argument(
        "bad_dir",
        type=Path,
        help="Path to directory where we want to sort images deemed bad. Can"
             " already exist, will overwrite images there if names match."
    )
    parser.add_argument(
        "-f", "--filetype",
        help="Filetype for images, will be used in glob. For example: jpg/jpeg/png",
        default="jpg",
    )
    parser.add_argument(
        "-r", "--recompute",
        help="Whether to recompute scores even if pickle file exists",
        action="store_true",
    )
    parser.add_argument(
        "-p", "--plot",
        help="Whether to plot results",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save",
        help="Save images to the folders",
        action="store_true",
    )
    parser.add_argument(
        "-t", "--threshold",
        help="Set the lap2 threshold for blurry images. If you don't know what"
             " a good value is, run this without setting the value and you'll"
             " get an interactive process.",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    assert args.image_dir.is_dir(), \
        f"{args.image_dir} needs to be a directory"

    if not args.good_dir.is_dir(): args.good_dir.mkdir()
    if not args.bad_dir.is_dir(): args.bad_dir.mkdir()

    assert args.image_dir.absolute() != args.good_dir.absolute()
    assert args.image_dir.absolute() != args.bad_dir.absolute()
    assert args.good_dir.absolute() != args.bad_dir.absolute()

    image_paths = sorted(list(args.image_dir.glob(f"*{args.filetype}")))
    assert len(image_paths) > 0, \
        f"{str(args.image_dir.absolute())}/*{args.filetype} produced no files"

    main(
        image_paths=image_paths,
        good_dir=args.good_dir,
        bad_dir=args.bad_dir,
        plot=args.plot,
        recompute=args.recompute,
        save_images=args.save,
        threshold=args.threshold,
    )
