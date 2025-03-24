

Here is the detailed README in Markdown format:

**Mask Segmentation Evaluation**
=====================================

**Table of Contents**
-----------------

- [**Mask Segmentation Evaluation**](#mask-segmentation-evaluation)
  - [**Table of Contents**](#table-of-contents)
  - [**Overview**](#overview)
  - [**Requirements**](#requirements)
  - [**Usage**](#usage)
    - [Running the Script](#running-the-script)
    - [Parser Options](#parser-options)
  - [**Dataset**](#dataset)
  - [**Output**](#output)
  - [**Notes**](#notes)
  - [**Troubleshooting**](#troubleshooting)

**Overview**
------------

This repository contains a Python script `main.py` that evaluates the performance of a mask segmentation algorithm on a dataset of images and masks. The script uses OpenCV and scikit-learn libraries to load images, apply mask segmentation, and calculate Intersection over Union (IoU) scores.

The script is designed to be flexible and can be used to evaluate different mask segmentation algorithms by modifying the `segmentation` function. The script also provides options for visualizing the results and calculating IoU scores for a random sample of images.

**Requirements**
---------------

* Python 3.6+
* OpenCV 4.5+
* scikit-learn 0.24+
* NumPy 1.20+
* Matplotlib 3.4+

**Usage**
-----

### Running the Script

To run the script, navigate to the directory containing `main.py` and execute the following command:
```bash
python main.py
```
This will run the script in the default mode, processing all images in the dataset.

### Parser Options

The script uses the `argparse` library to parse command-line arguments. The following options are available:

* `--mode`: Choose the processing mode:
	+ `all`: Process all images in the dataset (default)
	+ `sample`: Process a random sample of images (see `--num` option)
* `--num`: Specify the number of images to process in sample mode (default: 50)
* `--visualize`: Specify the number of images to visualize (default: 25)


Example usage:
```bash
python main.py --mode sample --num 100 --visualize 50
```
This will process a random sample of 100 images, visualize the results for 50 images, and use a random seed of 123.

**Dataset**
------------

The script assumes that the dataset is stored in a directory named `MSFD` containing subdirectories for images and masks. The directory structure should be as follows:
```bash
MSFD/
images/
image1.jpg
image2.jpg
...
masks/
mask1.png
mask2.png
...
```
The script will load images and masks from these directories and apply mask segmentation.

**Output**
----------

The script will output the following:

* IoU scores for each image
* Average IoU score for all images
* Visualization of the results for the specified number of images (if `--visualize` option is used)

The output will be saved in a directory named `results` created in the current working directory.

**Notes**
-------

* The script assumes that the images and masks are stored in the same directory structure as the dataset.
* The script uses the `jaccard_score` function from scikit-learn to calculate IoU scores.
* The script uses OpenCV to load and process images.
* The script uses Matplotlib to visualize the results.
* The script uses NumPy to perform numerical computations.
* The script uses Python's built-in `random` module to generate random numbers.

**Troubleshooting**
------------------

* If the script fails to load images or masks, check that the dataset directory structure is correct.
* If the script fails to calculate IoU scores, check that the segmentation function is correct.
* If the script fails to visualize results, check that Matplotlib is installed correctly.
