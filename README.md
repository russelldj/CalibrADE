# CalibrADE
## Accurate end-to-end camera calibration from redundant images
Developed by Aarrushi Shandilya, David Russell, and Eric Schneider for 16-811: Math Fundamentals for Robotics, Fall 2021, CMU


# Setup
Download the data from [this repo](https://github.com/YoniChechik/AI_is_Math/tree/master/c_07_camera_calibration/images) and put it in `./data`.

There's two magic things at play here, [poetry](https://python-poetry.org/) and [dvc](https://dvc.org/). Poetry is used to manage dependency versions but you can do it yourself. DVC is used to manage data, but for now you can just download it yourself from [here](https://github.com/YoniChechik/AI_is_Math/tree/master/c_07_camera_calibration/images) because I havn't set up a DVC remote yet.

# Quickstart
You can run the only functional code with `python calibrade/opencv_intro.py`.

# License
Distributed under the MIT license