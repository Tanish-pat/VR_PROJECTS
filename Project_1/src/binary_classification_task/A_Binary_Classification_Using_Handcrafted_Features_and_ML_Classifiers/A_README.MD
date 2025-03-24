# **A Binary Classification Using Handcrafted Features and ML Classifiers**

## **Table of Contents**
- [**A Binary Classification Using Handcrafted Features and ML Classifiers**](#a-binary-classification-using-handcrafted-features-and-ml-classifiers)
  - [**Table of Contents**](#table-of-contents)
- [**A Binary Classification Using Handcrafted Features and ML Classifiers**](#a-binary-classification-using-handcrafted-features-and-ml-classifiers-1)
  - [**1. Introduction**](#1-introduction)
  - [**2. Feature Extraction**](#2-feature-extraction)
    - [**2.1 Histogram of Oriented Gradients (HOG)**](#21-histogram-of-oriented-gradients-hog)
    - [**2.2 Color Histogram**](#22-color-histogram)
    - [**2.3 Local Binary Patterns (LBP)**](#23-local-binary-patterns-lbp)
    - [**2.4 Edge Histogram**](#24-edge-histogram)
  - [**3. Google Colab for Accelerated Model Training**](#3-google-colab-for-accelerated-model-training)
    - [**Colab Workflow**](#colab-workflow)
  - [**4. Model Evaluation on Local Machine**](#4-model-evaluation-on-local-machine)
    - [**Final Model Results**](#final-model-results)
  - [**5. Model Reports and Confusion Matrices**](#5-model-reports-and-confusion-matrices)
    - [**5.1 SVM**](#51-svm)
      - [**Classification Report:**](#classification-report)
      - [**Confusion Matrix:**](#confusion-matrix)
    - [**5.2 MLP**](#52-mlp)
      - [**Classification Report:**](#classification-report-1)
      - [**Confusion Matrix:**](#confusion-matrix-1)
    - [**5.3 XGBoost**](#53-xgboost)
      - [**Classification Report:**](#classification-report-2)
      - [**Confusion Matrix:**](#confusion-matrix-2)
    - [**5.4 RandomForest**](#54-randomforest)
      - [**Classification Report:**](#classification-report-3)
      - [**Confusion Matrix:**](#confusion-matrix-3)
- [**Conclusion**](#conclusion)

---

# **A Binary Classification Using Handcrafted Features and ML Classifiers**

## **1. Introduction**
This project focuses on binary classification for mask detection using handcrafted feature extraction techniques combined with machine learning models. The goal is to classify images into two categories: "with_mask" and "without_mask." We employ a robust feature extraction pipeline and evaluate multiple classifiers, including SVM, MLP, XGBoost, and RandomForest.

The workflow involves:
- Extracting features from images using HOG, Color Histograms, LBP, and Edge Histograms.
- Training multiple machine learning models for classification.
- Using Google Colab for accelerated model training.
- Performing final model evaluation on a local machine.

**Directory Structure**
```
C:.
│   A_README.MD
│   Colab_USAGE_ML.ipynb
│   feature_extraction.py
│   main.ipynb
│
├───enhanced_features
│       X.npy
│       y.npy
│
└───saved_models
    │   accuracy_log.txt
    │   MLP.pth
    │   MLP_report.txt
    │   RandomForest.pkl
    │   RandomForest_report.txt
    │   SVM.pkl
    │   SVM_report.txt
    │   XGBoost.pkl
    │   XGBoost_report.txt
    │
    └───plots
            MLP_confusion_matrix.png
            RandomForest_confusion_matrix.png
            SVM_confusion_matrix.png
            XGBoost_confusion_matrix.png
```

---

## **2. Feature Extraction**
We extract four types of handcrafted features from images to capture different aspects of the data:

### **2.1 Histogram of Oriented Gradients (HOG)**
HOG captures the structure and texture of an image by computing gradient orientations in localized regions. It is useful for detecting edges and shapes.

**Implementation:**
- Convert image to grayscale.
- Compute gradients in the x and y directions.
- Divide the image into small cells and compute gradient histograms.
- Normalize blocks of histograms for better invariance to lighting and contrast changes.

### **2.2 Color Histogram**
Color histograms capture the distribution of colors in an image, making them useful for distinguishing different object categories.

**Implementation:**
- Convert image to the HSV color space.
- Compute histograms for the H, S, and V channels.
- Normalize and concatenate the histograms.

### **2.3 Local Binary Patterns (LBP)**
LBP is a texture descriptor that encodes local patterns in an image based on pixel intensity differences.

**Implementation:**
- Convert image to grayscale.
- Compute LBP for each pixel using a circular neighborhood.
- Create a histogram of LBP values and normalize it.

### **2.4 Edge Histogram**
Edge histograms capture the distribution of edge intensities in an image.

**Implementation:**
- Convert image to grayscale.
- Apply Canny edge detection.
- Compute a histogram of detected edge intensities.

The final feature vector is obtained by concatenating HOG, Color Histogram, LBP, and Edge Histogram features.

---

## **3. Google Colab for Accelerated Model Training**
Since training multiple machine learning models can be computationally expensive, we utilized Google Colab to leverage free GPU resources.

### **Colab Workflow**
- Load extracted features from `enhanced_features/X.npy` and `enhanced_features/y.npy`.
- Train four models: SVM, MLP, XGBoost, and RandomForest.
- Save trained models in `saved_models/` for later evaluation.

Using Colab allowed us to train the MLP model efficiently using PyTorch with GPU acceleration. It also enabled parallel execution of hyperparameter tuning for SVM, XGBoost, and RandomForest.

---

## **4. Model Evaluation on Local Machine**
Once the models were trained in Colab, they were downloaded and evaluated in `main.ipynb`. The evaluation included:
- Loading test data (`X_test` and `y_test`).
- Computing accuracy, precision, recall, and F1-score.
- Generating confusion matrices.

### **Final Model Results**
| Model         | Accuracy |
|--------------|----------|
| **SVM**      | **93.87%** |
| **MLP**      | **93.25%** |
| **XGBoost**  | **92.64%** |
| **RandomForest** | **90.06%** |

**Observations:**
- SVM achieved the highest accuracy (93.87%), making it the best-performing model.
- MLP performed similarly well, benefiting from deep learning’s ability to capture complex patterns.
- XGBoost performed slightly worse, possibly due to its sensitivity to handcrafted feature distributions.
- RandomForest had the lowest accuracy, likely due to its reliance on feature randomness and decision tree limitations.

The saved models and evaluation reports can be found in the `saved_models/` directory.

---

---

## **5. Model Reports and Confusion Matrices**

Each trained model has a detailed classification report and a corresponding confusion matrix.

### **5.1 SVM**
#### **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.96      0.93      0.94       446
           1       0.91      0.95      0.93       369

    accuracy                           0.94       815
   macro avg       0.94      0.94      0.94       815
weighted avg       0.94      0.94      0.94       815

```
#### **Confusion Matrix:**

  ![SVM Confusion Matrix](/Project_1/src/binary_classification_task/A_Binary_Classification_Using_Handcrafted_Features_and_ML_Classifiers/saved_models/plots/SVM_confusion_matrix.png)

### **5.2 MLP**
#### **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.94      0.94      0.94       446
           1       0.92      0.93      0.93       369

    accuracy                           0.93       815
   macro avg       0.93      0.93      0.93       815
weighted avg       0.93      0.93      0.93       815
```
#### **Confusion Matrix:**

  ![MLP Confusion Matrix](/Project_1/src/binary_classification_task/A_Binary_Classification_Using_Handcrafted_Features_and_ML_Classifiers/saved_models/plots/MLP_confusion_matrix.png)

### **5.3 XGBoost**
#### **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.93      0.93      0.93       446
           1       0.92      0.92      0.92       369

    accuracy                           0.93       815
   macro avg       0.93      0.93      0.93       815
weighted avg       0.93      0.93      0.93       815
```
#### **Confusion Matrix:**

  ![XGBoost Confusion Matrix](/Project_1/src/binary_classification_task/A_Binary_Classification_Using_Handcrafted_Features_and_ML_Classifiers/saved_models/plots/XGBoost_confusion_matrix.png)

### **5.4 RandomForest**
#### **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.89      0.93      0.91       446
           1       0.91      0.87      0.89       369

    accuracy                           0.90       815
   macro avg       0.90      0.90      0.90       815
weighted avg       0.90      0.90      0.90       815
```
#### **Confusion Matrix:**

  ![RandomForest Confusion Matrix](/Project_1/src/binary_classification_task/A_Binary_Classification_Using_Handcrafted_Features_and_ML_Classifiers/saved_models/plots/RandomForest_confusion_matrix.png)

---

# **Conclusion**

Among the four models evaluated, **SVM achieved the highest accuracy (93.87%)**, making it the best-performing classifier for this binary classification task. **MLP (93.25%) and XGBoost (92.64%)** followed closely, showing competitive performance, while **RandomForest (90.06%)** had the lowest accuracy.

The results indicate that **SVM effectively leveraged the handcrafted features (HOG, LBP, Color Histograms, and Edge Histograms)** to achieve superior classification performance. The slight drop in MLP and XGBoost performance suggests that while deep learning and boosting methods are powerful, the handcrafted features might be more naturally suited for linear separability, which benefits SVM.

In summary, **SVM is the most optimal model for this feature extraction approach**, offering the best balance between accuracy and computational efficiency.