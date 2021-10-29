# CalibrADE
## Accurate end-to-end camera calibration from redundant images
Developed by Aarrushi Shandilya, David Russell, and Eric Schneider for 16-811: Math Fundamentals for Robotics, Fall 2021, CMU


# Setup

I have tried to set up [dvc](https://dvc.org/) to manage the data. The first step is to install `dvc` with `pip install dvc` or an analagous command. Then you *should* be able to run `dvc pull` within the repository and have it grab the data for you. There's a chance, however, that there's permission issues for other people trying to do that. If DVC doesn't work, you can just download `left*jpg` images from [here](https://github.com/opencv/opencv/blob/master/samples/data) and place them in `data/opencv_examples`.

I'm trying to manage dependencies with [Poetry](https://python-poetry.org/). I've found it a little tricky to use, so if you don't want to learn it, you can install the dependencies yourself. At this point they should just be `opencv` and `matplotlib`.

```
[For linux]
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
cd /path/to/CalibrADE/
poetry install  # Run this once at the beginning
poetry shell    # Run this every time you want to run stuff
```

It's also pip installable [![PyPI version](https://badge.fury.io/py/calibrade.svg)](https://badge.fury.io/py/calibrade) with `pip install calibrade`.

# DVC
```
dvc add <files>
run the git command that shows up
git push
dvc push
```

# Quickstart
You can run a simple OpenCV control point detection script with
```
python calibrade/opencv_intro.py
```

[Work in progress]
The pruning step can be run with these arguments. `--recompute` and `--threshold` may also be helpful as you play with it.
```
python calibrade/prune.py data/ good/ bad/ --filetype jpeg --plot --save
```

# License
Distributed under the MIT license