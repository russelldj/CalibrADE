"""
calibrate_intrinsics.py and run_repeated_calibrations.sh produce a series of
pickle files in directories where images are found (a.k.a. very scattered).
The filenames will have the pattern
    f"randomrun_{len(sampled_paths)}samples_{timestamp()}.pickle"
"""
import argparse
from collections import defaultdict
from glob import glob
from matplotlib import patches, pyplot
import numpy
from pathlib import Path
import pickle


def id_from_path(path):
    """From a path to a sampled pickle file, extract an ID for that run.
    E.g. from this path
        '../data/images/big_board/eric_phone/PXL_20211101_190909831/randomrun_4samples_1637106542543904.pickle'
    the relevant ID is
        big_board/eric_phone/
    """
    # Strip off early stuff
    path = path.split("data/images/")[-1]
    path = "/".join(path.split("/")[:2])
    return path


class Variability:
    """Holder class for certain variables"""
    def __init__(self, fx, fy, cx, cy, distortion0, distortion1, distortion2,
                 distortion3, distortion4, sharpness):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion0 = distortion0
        self.distortion1 = distortion1
        self.distortion2 = distortion2
        self.distortion3 = distortion3
        self.distortion4 = distortion4
        self.sharpness = sharpness


def extract_variability_data(all_pickles):
    by_id = defaultdict(lambda: defaultdict(list))
    sharpness_options = set()
    for file in all_pickles:
        id_name = id_from_path(file)
        with open(file, "rb") as handle:
            data = pickle.load(handle)
        num_sampled = len(data["sampled_paths"])

        # These were included by accident, weed them out
        if num_sampled == 5:
            continue

        # Capture the possible sharpnesses, with a default of 0
        if "sharpness_ratio" in data:
            sharpness = data["sharpness_ratio"]
        else:
            sharpness = 0
        sharpness_options.add(sharpness)

        by_id[id_name][num_sampled].append(Variability(
            fx=data["mtx"][0, 0],
            fy=data["mtx"][1, 1],
            cx=data["mtx"][0, 2],
            cy=data["mtx"][1, 2],
            distortion0=data["dist"][0, 0],
            distortion1=data["dist"][0, 1],
            distortion2=data["dist"][0, 2],
            distortion3=data["dist"][0, 3],
            distortion4=data["dist"][0, 4],
            sharpness=sharpness,
        ))
    return by_id, sharpness_options


def variablity_vs_samples(all_pickles, figuredir, suppress_outliers,
                          show_plot, specific_sharpness):
    by_id, sharpness_options = extract_variability_data(all_pickles)

    # Override the discovered sharpnesses if requested:
    if specific_sharpness is not None:
        sharpness_options = {specific_sharpness}

    for id_name, values in by_id.items():
        variables = [
            "fx", "fy", "cx", "cy", "skip", "distortion0", "distortion1",
            "distortion2", "distortion3", "distortion4"
        ]
        # variables = ["fy", "cy"]
        # variables = ["fy", "distortion1"]
        # variables = ["fy", "cy", "skip", "distortion4"]
        # variables = ["fx", "fy", "cy", "distortion0"]
        height = int(numpy.ceil(len(variables) / 2))
        figure, axes = pyplot.subplots(height, 2, figsize=(14, 10))
        for i, variable in enumerate(variables):
            if len(axes.shape) == 2:
                axis = axes[i % height, i // height]
                axes[-1, 0].set_xlabel("Number of randomly sampled images")
                axes[-1, 1].set_xlabel("Number of randomly sampled images")
            else:
                axis = axes[i]
                axes[0].set_xlabel("Number of randomly sampled images")
                axes[1].set_xlabel("Number of randomly sampled images")

            if variable == "skip":
                axis.legend(*zip(*labels))
                continue

            labels = []
            def add_label(violin, label):
                color = violin["bodies"][0].get_facecolor().flatten()
                labels.append((patches.Patch(color=color), label))

            samples = numpy.array(sorted(values.keys()))
            axis.set_ylabel(variable)
            axis.set_xticks(samples)

            radius = len(sharpness_options) // 2
            for sharpness, offset in zip(sorted(sharpness_options),
                                         numpy.arange(-radius, radius+1) * 1.25):
                chosen_samples = []
                dataset = []
                for sample in samples:
                    row = [
                        getattr(element, variable)
                        for element in values[sample]
                        if numpy.isclose(element.sharpness, sharpness)
                    ]
                    if len(row) > 0:
                        chosen_samples.append(sample + offset)
                        dataset.append(row)
                add_label(
                    axis.violinplot(dataset=dataset,
                                    positions=chosen_samples,
                                    widths=2.5-(1.0*radius),
                                    showextrema=not suppress_outliers),
                    f"Drop blurriest {sharpness * 100:.0f}%",
                )

        figure.suptitle(f"Variability for {id_name}")
        if show_plot:
            pyplot.show()
        else:
            pyplot.savefig(figuredir.joinpath(f"variability_{id_name.replace('/', '_')}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualize_repeated_calibrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datadir",
        type=Path,
        help="Directory that all images are stored downstream of",
    )
    parser.add_argument(
        "figuredir",
        type=Path,
        help="Directory where created figures should be saved",
    )
    parser.add_argument(
        "-o",
        "--suppress-outliers",
        action="store_true",
        help="Plot violin plots without outliers",
    )
    parser.add_argument(
        "-p",
        "--show-plot",
        action="store_true",
        help="Show plots to the user instead of saving the figure (blocking)",
    )
    parser.add_argument(
        "-s",
        "--specific-sharpness",
        type=float,
        default=None,
        help="Limit plot to a specific sharpness",
    )
    args = parser.parse_args()
    all_pickles = glob(str(args.datadir.joinpath("**/randomrun*pickle")),
                       recursive=True)
    variablity_vs_samples(
        all_pickles=all_pickles,
        figuredir=args.figuredir,
        suppress_outliers=args.suppress_outliers,
        show_plot=args.show_plot,
        specific_sharpness=args.specific_sharpness,
    )
