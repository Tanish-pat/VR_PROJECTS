# **Comparative Analysis: Traditional Machine Learning vs. CNN-Based Mask Detection**

This document presents a **comprehensive comparison** of the accuracy achieved by different models in both **Part A (Traditional ML Models)** and **Part B (CNN-Based Deep Learning Models)** for the **mask detection task**. We analyze the **best-performing models**, **hyperparameters**, and key insights that explain the differences in performance.
## **🚀 Steps to Run**

You can run any of the following Python files and Jupyter notebooks in Part A and Part B:

### **Part A:**
- **Python Files:**
  - `feature_extraction.py`
  - `Colab_USAGE_ML.ipynb` (to be run in Colab for training)
  - `main.ipynb` (to evaluate models)

### **Part B:**
- **Python Files:**
  - `Colab_USAGE_CNN.ipynb` (to be run in Colab for training larger models)
  - `main.ipynb` (to evaluate models)

### **Important Note:**
Do not run the Colab files locally; they are designed for execution in the Google Colab environment, which provides the necessary GPU resources and library configurations.
---

## 📌 **Table of Contents**
1. [Overview of Model Performance](#1-overview-of-model-performance)
2. [Best Model in Each Category](#2-best-model-in-each-category)
3. [Hyperparameters Comparison](#3-hyperparameters-comparison)
4. [Key Observations](#4-key-observations)
5. [Conclusion: Which Approach is Better?](#5-conclusion-which-approach-is-better)

---

## **Directory Structure**
```
C:.
├───COMPARISON_README.MD
├───A_Binary_Classification_Using_Handcrafted_Features_and_ML_Classifiers
│   ├───A_README.MD
│   ├───enhanced_features
│   └───saved_models
│       └───plots
└───B_Binary_Classification_Using_CNN
    ├───B_README.MD
    ├───Advanced_Analysis
    │   └───snapshots
    │       ├───histories
    │       ├───models
    │       └───plots
    └───Normal_Analysis
        ├───cnn_models
        └───cnn_processed_data
```



## **1️⃣ Overview of Model Performance**

Below is a **summary of the validation accuracies** achieved by different models across both approaches:

### **1.1 Part A: Traditional Machine Learning Models**
| Model          | Validation Accuracy |
|---------------|---------------------|
| **SVM**       | **93.87%**  |
| **MLP**       | 93.25%  |
| **XGBoost**   | 92.64%  |
| **RandomForest** | 90.06%  |

### **1.2 Part B: CNN-Based Deep Learning Models**
#### **B.1 Normal CNN Models**
| Model           | Validation Accuracy |
|---------------|---------------------|
| **ReLU + Adam** | **96.70%**  |
| **Tanh + Adam** | 96.58%  |
| **ReLU + SGD**  | 90.84%  |

#### **B.2 Advanced CNN Models**
| Model             | Validation Accuracy |
|------------------|---------------------|
| **Baseline CNN** | **97.68%**  |
| **VGG-like CNN** | 96.09%  |
| **ResNet-like CNN** | 95.48%  |
| **MobileNet**     | 🚨 **43.83%**  |

🔍 **Quick Insights:**
- **Best Traditional ML Model** → **SVM (93.87%)**
- **Best Normal CNN Model** → **ReLU + Adam (96.70%)**
- **Best Advanced CNN Model** → **Baseline CNN (97.68%)**
- **Best Overall Model** → **Baseline CNN (97.68%)**

---

## **2️⃣ Best Model in Each Category**

### **Traditional ML (Part A) Winner: SVM (93.87%)**
- **Strengths**: Works well with handcrafted feature extraction (HOG, LBP, Color Histogram).
- **Limitations**: Struggles with high-dimensional, complex patterns that CNNs can learn automatically.

### **CNN Normal Analysis (Part B) Winner: ReLU + Adam (96.70%)**
- **Strengths**: Optimized activation function and optimizer helped achieve superior accuracy.
- **Limitations**: Still not as strong as deeper CNNs in the Advanced Analysis.

### **CNN Advanced Analysis (Part B) Winner: Baseline CNN (97.68%)**
- **Strengths**: Balanced depth, good generalization, minimal overfitting.
- **Limitations**: Other deep models (VGG, ResNet) overfitted slightly, while MobileNet failed.

🚀 **Final Winner: Baseline CNN (97.68%)**, outperforming both traditional ML and simpler CNNs.

---

## **3️⃣ Hyperparameters Comparison**

| Approach         | Feature Extraction | Architecture | Optimizer | Epochs | Batch Size | Best Accuracy |
|-----------------|--------------------|-------------|----------|--------|------------|--------------|
| **Part A (ML)** | HOG, LBP, Color Hist. | SVM, MLP, XGBoost | N/A | N/A | N/A | **93.87% (SVM)** |
| **Part B Normal** | None (Raw Images) | Simple CNN | Adam/SGD | 25 | 64 | **96.70% (ReLU + Adam)** |
| **Part B Advanced** | None (Raw Images) | Deeper CNNs | Adam | 25 | 64 | **97.68% (Baseline CNN)** |

📝 **Key Takeaways:**
- **Traditional ML required handcrafted feature extraction**, whereas **CNNs learned features automatically** from raw images.
- **Deeper CNNs had better accuracy**, but very deep models (VGG, ResNet) started overfitting slightly.
- **Batch Size and Epochs were constant** across CNNs to ensure a fair comparison.

---

## **4️⃣ Key Observations**

### **4.1 Why Did CNNs Outperform Traditional ML?**
1. **Automatic Feature Learning** → CNNs extract hierarchical features (edges, textures, complex patterns) that ML models cannot capture manually.
2. **End-to-End Training** → CNNs train directly on raw images, avoiding errors introduced by feature engineering.
3. **Deep Architectures Improve Learning** → The best model (Baseline CNN) used stacked convolutional layers for richer feature representation.

### **4.2 Why Did Some CNNs Perform Worse?**
- **ReLU + SGD (Normal) had a lower accuracy (90.84%)** due to slower learning dynamics.
- **MobileNet (43.83%) failed** due to missing pretrained weights and improper initialization.
- **VGG-like and ResNet-like had overfitting issues**, suggesting that their depth was excessive for this dataset.

### **4.3 Could Traditional ML Work Better?**
- **If dataset size was smaller**, ML models might be preferable due to their lower computational cost.
- **With feature engineering improvements**, ML models could close the gap but likely not surpass CNNs.

---

## **5️⃣ Conclusion: Which Approach is Better?**

| **Metric** | **Traditional ML (Part A)** | **CNN (Part B Normal)** | **CNN (Part B Advanced)** |
|------------|----------------------------|-------------------------|---------------------------|
| **Best Accuracy** | 93.87% (SVM) | 96.70% (ReLU + Adam) | **97.68% (Baseline CNN)** |
| **Feature Extraction** | HOG, LBP, Color Histogram | None (learned features) | None (learned features) |
| **Training Time** | Fast | Moderate | **High (longer training)** |
| **Generalization** | Moderate | High | **Very High** |
| **Computational Cost** | Low | Medium | **High (requires GPU)** |

### **🏆 Final Verdict: CNN-Based Models are Superior**
- **CNNs significantly outperformed Traditional ML**, with **Baseline CNN (97.68%) being the best model overall**.
- **Traditional ML is still viable** for smaller datasets or low-compute scenarios.
- **Feature extraction is a key limitation** of Traditional ML, while CNNs adapt automatically.

📌 **Recommendation:** Use **Baseline CNN for production** unless compute resources are limited, in which case **SVM could be a lightweight alternative**.

---

# This completes the accuracy comparison across all models in **Part A (Traditional ML)** and **Part B (CNNs)**.