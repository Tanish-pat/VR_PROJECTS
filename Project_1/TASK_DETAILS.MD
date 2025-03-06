# **Face Mask Detection, Classification, and Segmentation**

## **Project Details**
- **Total Marks:** 15
- **Deadline:** 24th March, 2025

## **Objective**
Develop a computer vision solution to classify and segment face masks in images. The project involves using **handcrafted features** with **machine learning classifiers** and **deep learning techniques** to perform classification and segmentation.

---

## **Dataset**
- **Face Mask Detection Dataset:** Labeled dataset containing images of people with and without face masks.
  🔗 [https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
- **Masked Face Segmentation Dataset:** Includes ground truth mask segmentations.
  🔗 [https://github.com/sadjadrz/MFSD](https://github.com/sadjadrz/MFSD)
---

This ensures the links are visible even in plain text formats. Let me know if you need further adjustments!



## **Tasks and Marks Distribution**

### **A. Binary Classification Using Handcrafted Features and ML Classifiers (4 Marks)**
- ✔ Extract handcrafted features from the dataset.
- ✔ Train and evaluate at least **two machine learning classifiers** (e.g., **SVM, Neural Network**) to classify faces as **"with mask"** or **"without mask."**
- ✔ Report and compare the accuracy of the classifiers.

### **B. Binary Classification Using CNN (3 Marks)**
- ✔ Design and train a **Convolutional Neural Network (CNN)** for binary classification.
- ✔ Experiment with **hyperparameter variations** (e.g., learning rate, batch size, optimizer, activation function in the classification layer) and report results.
- ✔ Compare the CNN's performance with the ML classifiers.

### **C. Region Segmentation Using Traditional Techniques (3 Marks)**
- ✔ Implement a **region-based segmentation method** (e.g., **thresholding, edge detection**) for segmenting mask regions in faces identified as **"with mask."**
- ✔ Visualize and evaluate the segmentation results.

### **D. Mask Segmentation Using U-Net (5 Marks)**
- ✔ Train a **U-Net model** for precise segmentation of mask regions in images.
- ✔ Compare the **U-Net performance** with the traditional segmentation method using **metrics like IoU or Dice score.**

---

## **Submission Guidelines**

- 📌 **Repository Name:** `VR_Project1_[YourName]_[YourRollNo]`
- 📌 **Code Submission Format:**
  - **Binary Classification Tasks** (Handcrafted Features & CNN) → Submit as a single script or notebook.
  - **Segmentation Tasks** (Traditional Techniques & U-Net) → Submit as a separate script or notebook.

---

### **README File Requirements**
Your README must serve as a **detailed project report** and include:

1️⃣ **Introduction**
   - Brief overview of project objectives.

2️⃣ **Dataset**
   - Dataset details, including source and structure.

3️⃣ **Methodology**
   - Step-by-step explanation of **feature extraction, model training, and segmentation techniques.**

4️⃣ **Hyperparameters & Experiments**
   - List and describe hyperparameters used for **CNN and U-Net models**, along with variations and results.

5️⃣ **Results**
   - Report evaluation metrics (**accuracy, IoU, Dice score, etc.**) and compare different approaches.

6️⃣ **Observations & Analysis**
   - Summarize insights gained from the results, challenges faced, and how they were addressed.

7️⃣ **How to Run the Code**
   - Provide step-by-step instructions to execute the scripts.

⚠ **Ensure that your code runs smoothly without requiring manual intervention and that all outputs are clearly labeled.**
---