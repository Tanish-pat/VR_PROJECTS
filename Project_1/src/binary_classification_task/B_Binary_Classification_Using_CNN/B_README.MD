# Mask Detection using CNNs (Part B)

## Table of Contents
- [Mask Detection using CNNs (Part B)](#mask-detection-using-cnns-part-b)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Dataset \& Preprocessing](#2-dataset--preprocessing)
  - [3. Normal Analysis](#3-normal-analysis)
    - [3.1 Model Architecture](#31-model-architecture)
    - [3.2 Training Process](#32-training-process)
    - [3.3 Evaluation](#33-evaluation)
  - [4. Advanced Analysis](#4-advanced-analysis)
    - [4.1 Motivation for Advanced Analysis](#41-motivation-for-advanced-analysis)
    - [4.2 Model Architectures](#42-model-architectures)
    - [4.3 Training Process](#43-training-process)
    - [4.4 Evaluation](#44-evaluation)
  - [5. Differences Between Models](#5-differences-between-models)
    - [5.1 Model Complexity \& Architecture](#51-model-complexity--architecture)
    - [5.2 Computational Efficiency](#52-computational-efficiency)
    - [5.3 Performance \& Accuracy](#53-performance--accuracy)
    - [5.4 Takeaways](#54-takeaways)
  - [6. Model Performance Visualization](#6-model-performance-visualization)
    - [6.1 Normal Analysis Model Accuracy Plots](#61-normal-analysis-model-accuracy-plots)
      - [📌 Accuracy Trends for Normal Analysis](#-accuracy-trends-for-normal-analysis)
    - [6.2 Advanced Analysis Model Accuracy Plots](#62-advanced-analysis-model-accuracy-plots)
      - [📌 Accuracy Trends for Advanced Analysis](#-accuracy-trends-for-advanced-analysis)
    - [6.3 Comparative Observations](#63-comparative-observations)
      - [📌 Conclusion: **More complex doesn’t always mean better.**](#-conclusion-more-complex-doesnt-always-mean-better)

---

## 1. Introduction

This project aims to classify images of faces as **"with mask"** or **"without mask"** using **Convolutional Neural Networks (CNNs)**. In **Part A**, we used **feature extraction with multiple machine learning models** (SVM, MLP, XGBoost, etc.), whereas in **Part B**, we focus on **deep learning-based classification** using CNNs.

We divide Part B into **two phases**:
- **Normal Analysis**: Uses smaller CNN architectures with different activation functions and optimizers.
- **Advanced Analysis**: Uses larger CNN models (including ResNet-like and MobileNet).

The goal is to **compare the effectiveness of simple vs. advanced CNN architectures** for mask detection.
**Directory Structure**
```
C:.
│   B_README.MD
│
├───Advanced_Analysis
│   │   Colab_USAGE_CNN.ipynb
│   │   main.ipynb
│   │
│   └───snapshots
│       │   hyperparams.json
│       │
│       ├───histories
│       │       baseline_cnn.json
│       │       mobilenet.json
│       │       resnet_like.json
│       │       vgg_like.json
│       │
│       ├───models
│       │       baseline_cnn.h5
│       │       mobilenet.h5
│       │       resnet_like.h5
│       │       vgg_like.h5
│       │
│       └───plots
│               baseline_cnn.png
│               mobilenet.png
│               resnet_like.png
│               vgg_like.png
│
└───Normal_Analysis
    │   main.ipynb
    │
    ├───cnn_models
    │       history_relu_adam.json
    │       history_relu_sgd.json
    │       history_tanh_adam.json
    │       model_relu_adam.keras
    │       model_relu_sgd.keras
    │       model_tanh_adam.keras
    │
    └───cnn_processed_data
            relu_adam.png
            relu_sgd.png
            tanh_adam.png
            X.npy
            y.npy
```
---

## 2. Dataset & Preprocessing

The dataset is preprocessed into **NumPy arrays** (`X.npy`, `y.npy`) for efficient loading and training. These files are used across both Normal and Advanced Analyses.

- **Image Size**: `(128, 128, 3)`
- **Classes**: `with_mask (1)` and `without_mask (0)`
- **Train-Test Split**: `80%-20%`

---

## 3. Normal Analysis

The **Normal Analysis** focuses on simple CNN models trained directly in **Jupyter Notebook**. The purpose is to test the effectiveness of CNNs using basic architectures before moving to advanced models.

### 3.1 Model Architecture

Three models were trained with varying activation functions and optimizers:
- **Model 1:** `ReLU + Adam`
- **Model 2:** `Tanh + Adam`
- **Model 3:** `ReLU + SGD`

All models follow a **3-layer CNN architecture**:
1. **Conv2D(32) → MaxPooling**
2. **Conv2D(64) → MaxPooling**
3. **Conv2D(128) → MaxPooling → Flatten → Dense(128) → Output Layer**

### 3.2 Training Process

- **Epochs**: `10`
- **Batch Size**: `32`
- **Loss Function**: `Binary Crossentropy`
- **Evaluation Metric**: `Accuracy`

### 3.3 Evaluation

| Model | Validation Accuracy |
|--------|---------------------|
| `ReLU + Adam`  | **96.70%** |
| `Tanh + Adam`  | **96.58%** |
| `ReLU + SGD`   | **90.84%** |

**Key Findings:**
✅ `ReLU + Adam` performed the best.
⚠️ `ReLU + SGD` performed the worst, likely due to **SGD's slow convergence**.

---

## 4. Advanced Analysis

### 4.1 Motivation for Advanced Analysis

Although **Normal Analysis** achieved high accuracy, its models were **simple** and might not generalize well on more diverse datasets. To improve, we explore **larger and more complex architectures** using **Google Colab (GPU acceleration)**.

### 4.2 Model Architectures

Four deep CNN models were trained:

1. **Baseline CNN** – Similar to Normal Analysis but deeper.
2. **VGG-like CNN** – Inspired by VGG architecture, uses **more layers**.
3. **ResNet-like CNN** – Introduces **residual connections** to prevent vanishing gradients.
4. **MobileNet** – Lightweight, designed for mobile applications.

### 4.3 Training Process

- **Epochs**: `25` (more than Normal Analysis for deeper models)
- **Batch Size**: `64` (higher for efficient GPU utilization)
- **Optimizer**: `Adam` for all models

### 4.4 Evaluation

| Model | Validation Accuracy | Validation Loss |
|---------------|---------------------|-----------------|
| **Baseline CNN** | **97.68%** | `0.1619` |
| **VGG-like CNN** | `96.09%` | `0.2767` |
| **ResNet-like CNN** | `95.48%` | `0.2748` |
| **MobileNet** | 🚨 `43.83%` | 🚨 `2.3555` |

---

## 5. Differences Between Models

The four CNN models used in Advanced Analysis—**Baseline CNN, VGG-like, ResNet-like, and MobileNet**—vary significantly in their architectures, complexity, and intended purpose. Below is a structured comparison highlighting their differences:

### 5.1 Model Complexity & Architecture

| Model | Depth | Key Features | Intended Benefit |
|--------------|------|--------------------------------|----------------|
| **Baseline CNN** | Shallow | 3 Conv layers, MaxPooling | Simple, fast training |
| **VGG-like CNN** | Deeper | 5 Conv layers, deeper feature extraction | Stronger feature learning |
| **ResNet-like CNN** | Deep with Residuals | Skip connections to prevent vanishing gradients | Improved gradient flow |
| **MobileNet** | Lightweight | Depthwise separable convolutions | Optimized for mobile devices |

- **Baseline CNN** is the simplest and follows a traditional architecture.
- **VGG-like CNN** is inspired by VGG networks and adds depth to improve feature extraction.
- **ResNet-like CNN** introduces **residual connections**, allowing deeper networks to train efficiently.
- **MobileNet** uses **depthwise separable convolutions** to make the model smaller and faster.

### 5.2 Computational Efficiency

| Model | Parameters | Training Time (Approx) | GPU Requirement |
|--------------|------------|------------------|-----------------|
| **Baseline CNN** | **~1.2M** | 🚀 Fastest | Low |
| **VGG-like CNN** | **~3.5M** | 🐢 Slower | Medium |
| **ResNet-like CNN** | **~2.9M** | ⚖️ Moderate | High |
| **MobileNet** | **~2.2M** | ⏳ Slow (unexpected) | Low (but failed) |

- **Baseline CNN** is computationally light and trains quickly.
- **VGG-like CNN** has more layers and parameters, increasing training time.
- **ResNet-like CNN** is deep but efficient due to residual connections.
- **MobileNet** should have been fast, but its training failed, possibly due to missing **pretrained weights**.

### 5.3 Performance & Accuracy

| Model | Validation Accuracy | Key Issue |
|--------------|----------------|----------------------|
| **Baseline CNN** | **97.68%** ✅ | Strong performance |
| **VGG-like CNN** | **96.09%** 🔻 | Overfitting risk |
| **ResNet-like CNN** | **95.48%** 🔻 | Similar to VGG-like |
| **MobileNet** | 🚨 **43.83%** ❌ | Training failed (likely weight issue) |

- **Baseline CNN performed the best**, proving that even a simple model can achieve high accuracy.
- **VGG-like and ResNet-like models did not significantly improve accuracy**, suggesting **diminishing returns with depth**.
- **MobileNet completely failed**, likely because it lacked pretrained weights, making it unable to extract meaningful features.

### 5.4 Takeaways

✅ **Simple CNNs can be very effective** – Baseline CNN outperformed deeper models.
⚠️ **Depth doesn’t always mean better performance** – Overfitting can be an issue in deeper networks.
❌ **Pretrained weights matter** – MobileNet needs them for effective feature extraction.

---

## 6. Model Performance Visualization

To better understand how each model performed, we visualize their training history. Accuracy plots for both **Normal Analysis** and **Advanced Analysis** provide insights into learning trends, overfitting, and generalization capabilities.

### 6.1 Normal Analysis Model Accuracy Plots

The Normal Analysis involved three models trained with different activation function and optimizer combinations:

- **ReLU + Adam**
- **ReLU + SGD**
- **Tanh + Adam**

#### 📌 Accuracy Trends for Normal Analysis

Stored in `cnn_processed_data/` as:

- ![ReLU + Adam](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Normal_Analysis/cnn_processed_data/relu_adam.png)
- ![ReLU + SGD](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Normal_Analysis/cnn_processed_data/relu_sgd.png)
- ![Tanh + Adam](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Normal_Analysis/cnn_processed_data/tanh_adam.png)

These plots highlight:

✅ **ReLU + Adam achieved stable convergence** with minimal overfitting.
🔻 **ReLU + SGD had fluctuations**, indicating some instability.
⚠️ **Tanh + Adam showed slower learning** but was stable.

### 6.2 Advanced Analysis Model Accuracy Plots

The Advanced Analysis trained four distinct CNN architectures:

- **Baseline CNN**
- **VGG-like CNN**
- **ResNet-like CNN**
- **MobileNet**

#### 📌 Accuracy Trends for Advanced Analysis

Stored in `snapshots/plots/` as:

| Baseline CNN | VGG-like |
|-------------|---------|
| ![Baseline CNN](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Advanced_Analysis/snapshots/plots/baseline_cnn.png) | ![VGG-like](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Advanced_Analysis/snapshots/plots/vgg_like.png) |

| ResNet-like | MobileNet |
|------------|---------|
| ![ResNet-like](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Advanced_Analysis/snapshots/plots/resnet_like.png) | ![MobileNet](/Project_1/src/binary_classification_task/B_Binary_Classification_Using_CNN/Advanced_Analysis/snapshots/plots/mobilenet.png) |
These plots reveal:

✅ **Baseline CNN maintained strong performance** throughout training.
⚠️ **VGG-like and ResNet-like models overfitted slightly** compared to the baseline.
❌ **MobileNet failed to learn** due to missing pretrained weights, causing it to collapse.

### 6.3 Comparative Observations

| Model Type | Best Performing Model | Overfitting Risk | Key Observation |
|------------|---------------------|-----------------|-----------------|
| **Normal Analysis** | ReLU + Adam | Low | Simpler networks generalize well. |
| **Advanced Analysis** | Baseline CNN | Low | Deeper networks didn't provide significant gains. |

#### 📌 Conclusion: **More complex doesn’t always mean better.**
- The **Baseline CNN (Advanced)** outperformed deeper models like **VGG-like and ResNet-like**.
- The **simpler models in Normal Analysis performed nearly as well as Advanced Analysis models**.
- **MobileNet failed**, reinforcing the importance of **pretrained weights** in transfer learning approaches.

---