# CalibrADE
## Accurate end-to-end camera calibration from redundant images
Developed by Aarrushi Shandilya, David Russell, and Eric Schneider for 16-811: Math Fundamentals for Robotics, Fall 2021, CMU


# Setup
The main dependencies of this project are `opencv` and `matplotlib`.

There's two magic tools in play here, [poetry](https://python-poetry.org/) and [dvc](https://dvc.org/). Poetry is used to manage dependency versions but you can do it yourself. DVC is used to manage data, but for now you can just download `left*jpg` images from [here](https://github.com/opencv/opencv/blob/master/samples/data) and place them in `data/opencv_examples` because I havn't set up a DVC remote yet.

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