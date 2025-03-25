# Mask Segmentation Using U-Net

## Table of Contents
- [Mask Segmentation Using U-Net](#mask-segmentation-using-u-net)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Installation and Setup](#installation-and-setup)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
  - [How to Run](#how-to-run)
    - [1. Train the Model](#1-train-the-model)
    - [2. Perform Inference and Evaluate the Model](#2-perform-inference-and-evaluate-the-model)
  - [Adding Test Images](#adding-test-images)
  - [Model Architecture](#model-architecture)
  - [Training Strategy](#training-strategy)
  - [Performance Evaluation](#performance-evaluation)
  - [Comparison with Traditional Methods](#comparison-with-traditional-methods)
  - [Inference and Results](#inference-and-results)
  - [Future Work](#future-work)

## Project Overview
This project implements a U-Net model for mask segmentation in images. The model is trained on a dataset of facial images with mask annotations and is compared against traditional segmentation techniques.

## Directory Structure
The repository is structured as follows:

```
D_UNET_method/
│── Saved_Model/
│   ├── Best_model.pth
│── msfd-unet-segmentation.ipynb
│── test_images/
│   ├── test_img_1.jpg
│   ├── test_img_2.jpg
│   ├── test_img_3.jpg
│── checkpoints/
```

- **Saved_Model/**: Contains the trained model weights.
- **msfd-unet-segmentation.ipynb**: Main Jupyter notebook for training and evaluation.
- **test_images/**: Directory for storing test images.
- **checkpoints/**: Contains intermediate model checkpoints.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone git@github.com:Tanish-pat/VR_PROJECTS.git
```
Navigate into this directory.

### 2. Create and Activate a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the Model
To train the model, open `msfd-unet-segmentation.ipynb` in Jupyter Notebook or run:
```bash
jupyter notebook msfd-unet-segmentation.ipynb
```

This script computes IoU and Dice Score metrics for the model.

### 2. Perform Inference and Evaluate the Model

Load your images into the test_images directory. (.jpg). Run the Notebook cell on inference



## Adding Test Images
To add new test images, place them inside the `test_images/` directory. Then, update the inference command:


## Model Architecture
The U-Net model consists of:
- An encoder with convolutional layers and max-pooling.
- A bottleneck connecting encoder and decoder.
- A decoder with up-convolutions and skip connections.
- A final layer producing a binary segmentation mask.

## Training Strategy
- **Loss Function**: Cross-Entropy + Dice Loss
- **Optimizer**: AdamW with learning rate scheduling
- **K-Fold Cross-Validation (K=3)**
- **Early Stopping**: Prevents overfitting

## Performance Evaluation
The model is evaluated using:
```math
IoU = \frac{TP}{TP + FP + FN}
```
```math
Dice = \frac{2TP}{2TP + FP + FN}
```
Final results:
- **Mean IoU**: 0.7958
- **Mean Dice Score**: 0.8858

## Comparison with Traditional Methods
Traditional segmentation methods like thresholding and edge detection yield an IoU of ~0.6, while U-Net achieves **0.7958**, proving its superiority.

## Inference and Results
To run inference on new images, go to the notebook and see the inference section
Example results:
Refer to the detailed documentation provided. 
## Future Work
- Experimenting with deeper architectures.
- Training on larger datasets.
- Implementing real-time inference for video streams.

---
Feel free to contribute to the project by submitting issues or pull requests!
