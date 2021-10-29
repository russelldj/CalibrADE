# CalibrADE
## Accurate end-to-end camera calibration from redundant images
Developed by Aarrushi Shandilya, David Russell, and Eric Schneider for 16-811: Math Fundamentals for Robotics, Fall 2021, CMU


# Setup
Download the data from [this repo](https://github.com/YoniChechik/AI_is_Math/tree/master/c_07_camera_calibration/images) and put it in `./data`.

There's two magic things at play here, [poetry](https://python-poetry.org/) and [dvc](https://dvc.org/). Poetry is used to manage dependency versions but you can do it yourself. DVC is used to manage data, but for now you can just download `left*jpg` images from [here](https://github.com/opencv/opencv/blob/master/samples/data) and place them in `data/opencv_examples` because I havn't set up a DVC remote yet.

# Quickstart
You can run the only functional code with `python calibrade/opencv_intro.py`.

[Work in progress]
The pruning step can be run with these arguments. `--recompute` and `--threshold` may also be helpful as you play with it.
```
python calibrade/prune.py data/ good/ bad/ --filetype jpeg --plot --save
```

# License
Distributed under the MIT license