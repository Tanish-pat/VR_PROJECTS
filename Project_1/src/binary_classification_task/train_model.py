import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load extracted features (from the folder)
# FEATURES_FOLDER = "testing_features"
FEATURES_FOLDER = "enhanced_second_features"
X = np.load(os.path.join(FEATURES_FOLDER, "X.npy"))
y = np.load(os.path.join(FEATURES_FOLDER, "y.npy"))
print("loaded")
print(X.shape)
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train SVM model
# svm_model = SVC(kernel="linear")
# svm_model.fit(X_train, y_train)
# print("svm trained")
# svm_pred = svm_model.predict(X_test)
# svm_acc = accuracy_score(y_test, svm_pred)

# print(f"SVM Accuracy: {svm_acc:.4f}")

# # SVM Classification Report and Confusion Matrix
# print("\nSVM Classification Report:")
# print(classification_report(y_test, svm_pred))

# # SVM Classification Report:
# #               precision    recall  f1-score   support

# #            0       0.92      0.90      0.91       446
# #            1       0.88      0.91      0.90       369

# #     accuracy                           0.91       815
# #    macro avg       0.90      0.91      0.90       815
# # weighted avg       0.91      0.91      0.91       815

# print("\nSVM Confusion Matrix:")
# svm_cm = confusion_matrix(y_test, svm_pred)
# sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues")
# plt.title("SVM Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------



# # Train Neural Network (MLP)
# mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000)
# mlp_model.fit(X_train, y_train)
# print("mlp trained")
# mlp_pred = mlp_model.predict(X_test)
# mlp_acc = accuracy_score(y_test, mlp_pred)
# print(f"MLP Accuracy: {mlp_acc:.4f}")

# # MLP Classification Report and Confusion Matrix
# print("\nMLP Classification Report:")
# print(classification_report(y_test, mlp_pred))

# print("\nMLP Confusion Matrix:")
# mlp_cm = confusion_matrix(y_test, mlp_pred)
# sns.heatmap(mlp_cm, annot=True, fmt="d", cmap="Greens")
# plt.title("MLP Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# # MLP Accuracy: 0.9399

# # MLP Classification Report:
# #               precision    recall  f1-score   support

# #            0       0.94      0.95      0.95       446
# #            1       0.93      0.93      0.93       369

# #     accuracy                           0.94       815
# #    macro avg       0.94      0.94      0.94       815
# # weighted avg       0.94      0.94      0.94       815




# ------------------------------------------------------------------------------------------------------------------------

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
    )
xgb_model.fit(X_train, y_train)

print("XGBoost trained")

# Predict with XGBoost
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# Classification Report
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# Confusion Matrix
print("\nXGBoost Confusion Matrix:")
xgb_cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# XGBoost trained
# XGBoost Accuracy: 0.9239

# XGBoost Classification Report:
#               precision    recall  f1-score   support

#            0       0.93      0.93      0.93       446
#            1       0.92      0.91      0.92       369

#     accuracy                           0.92       815
#    macro avg       0.92      0.92      0.92       815
# weighted avg       0.92      0.92      0.92       815

# ------------------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=None,    # No depth limit (can be adjusted to prevent overfitting)
    random_state=42,   # Ensures reproducibility
    n_jobs=-1          # Use all available CPU cores for faster training
)
rf_model.fit(X_train, y_train)

print("Random Forest trained")

# Predict with Random Forest
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
print("\nRandom Forest Confusion Matrix:")
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Random Forest trained
# Random Forest Accuracy: 0.9117

# Random Forest Classification Report:
#               precision    recall  f1-score   support

#            0       0.91      0.93      0.92       446
#            1       0.91      0.89      0.90       369

#     accuracy                           0.91       815
#    macro avg       0.91      0.91      0.91       815
# weighted avg       0.91      0.91      0.91       815

# ------------------------------------------------------------------------------------------------------------------------