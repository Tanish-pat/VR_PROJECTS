import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ Load Features ------------------
FEATURES_FOLDER = "enhanced_features"
X = np.load(os.path.join(FEATURES_FOLDER, "X.npy"))
y = np.load(os.path.join(FEATURES_FOLDER, "y.npy"))
print("Features Loaded:", X.shape, y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model save/load directory (Local system)
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------ Function to Train & Save Model ------------------
def train_and_save_model(model, model_name, train_progress=False):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"{model_name} model found, loading...")
        return joblib.load(model_path)

    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"{model_name} saved at {model_path}\n")
    return model


# ------------------ Train or Load Models with Live Progress ------------------

svm_model = train_and_save_model(
    SVC(kernel="linear", C=1.0, probability=True, verbose=True),  # SVM prints training progress
    "SVM"
)

mlp_model = train_and_save_model(
    MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1500, alpha=0.0001, verbose=True),  # MLP prints epochs
    "MLP"
)

xgb_model = train_and_save_model(
    xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        verbosity=1  # XGBoost shows training progress
    ),
    "XGBoost"
)

rf_model = train_and_save_model(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=1  # Shows tree-building progress
    ),
    "RandomForest"
)


# ------------------ Function to Evaluate Model ------------------
def evaluate_model(model, model_name):
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print(f"\n{model_name} Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return acc


# ------------------ Evaluate All Models ------------------
models = {
    "SVM": svm_model,
    "MLP": mlp_model,
    "XGBoost": xgb_model,
    "RandomForest": rf_model
}

accuracies = {name: evaluate_model(model, name) for name, model in models.items()}

# Print Comparison of Accuracies
print("\nModel Comparison:")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.4f}")

# List saved models
print("\nSaved Models in 'saved_models' Directory:")
for file in os.listdir(MODEL_DIR):
    print(file)

















# import numpy as np
# import os
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # ------------------ Load Features ------------------
# FEATURES_FOLDER = "enhanced_features"
# X = np.load(os.path.join(FEATURES_FOLDER, "X.npy"))
# y = np.load(os.path.join(FEATURES_FOLDER, "y.npy"))
# print("Features Loaded:", X.shape, y.shape)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model save/load directory
# MODEL_DIR = "saved_models"
# os.makedirs(MODEL_DIR, exist_ok=True)


# # ------------------ Function to Train & Save Model ------------------
# def train_and_save_model(model, model_name):
#     model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

#     # If model exists, load it
#     if os.path.exists(model_path):
#         print(f"{model_name} model found, loading...")
#         return joblib.load(model_path)

#     # Otherwise, train and save it
#     print(f"Training {model_name}...")
#     model.fit(X_train, y_train)
#     joblib.dump(model, model_path)
#     print(f"{model_name} saved at {model_path}")
#     return model


# # ------------------ Train or Load Models with Some Hyperparameters ------------------

# svm_model = train_and_save_model(
#     SVC(kernel="linear", C=1.0, probability=True),  # Regularization added
#     "SVM"
# )

# mlp_model = train_and_save_model(
#     MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1500, alpha=0.0001),  # L2 regularization
#     "MLP"
# )

# xgb_model = train_and_save_model(
#     xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="logloss",
#         n_estimators=200,  # More trees
#         learning_rate=0.1  # Explicitly set
#     ),
#     "XGBoost"
# )

# rf_model = train_and_save_model(
#     RandomForestClassifier(
#         n_estimators=100,
#         max_depth=10,  # Prevent deep trees from overfitting
#         random_state=42,
#         n_jobs=-1
#     ),
#     "RandomForest"
# )


# # ------------------ Function to Evaluate Model ------------------
# def evaluate_model(model, model_name):
#     print(f"\nEvaluating {model_name}...")
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"{model_name} Accuracy: {acc:.4f}")

#     # Classification Report
#     print(f"\n{model_name} Classification Report:")
#     print(classification_report(y_test, y_pred))

#     # Confusion Matrix
#     print(f"\n{model_name} Confusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"{model_name} Confusion Matrix")
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.show()

#     return acc


# # ------------------ Evaluate All Models ------------------
# models = {
#     "SVM": svm_model,
#     "MLP": mlp_model,
#     "XGBoost": xgb_model,
#     "RandomForest": rf_model
# }

# accuracies = {name: evaluate_model(model, name) for name, model in models.items()}

# # Print Comparison of Accuracies
# print("\nModel Comparison:")
# for name, acc in accuracies.items():
#     print(f"{name}: {acc:.4f}")











# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load extracted features (from the folder)
# FEATURES_FOLDER = "enhanced_features"
# X = np.load(os.path.join(FEATURES_FOLDER, "X.npy"))
# y = np.load(os.path.join(FEATURES_FOLDER, "y.npy"))
# print("loaded")
# print(X.shape)
# print(y.shape)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# # # SVM Classification Report:
# # #               precision    recall  f1-score   support

# # #            0       0.92      0.90      0.91       446
# # #            1       0.88      0.91      0.90       369

# # #     accuracy                           0.91       815
# # #    macro avg       0.90      0.91      0.90       815
# # # weighted avg       0.91      0.91      0.91       815

# print("\nSVM Confusion Matrix:")
# svm_cm = confusion_matrix(y_test, svm_pred)
# sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues")
# plt.title("SVM Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# # ------------------------------------------------------------------------------------------------------------------------



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

# # # MLP Accuracy: 0.9399

# # # MLP Classification Report:
# # #               precision    recall  f1-score   support

# # #            0       0.94      0.95      0.95       446
# # #            1       0.93      0.93      0.93       369

# # #     accuracy                           0.94       815
# # #    macro avg       0.94      0.94      0.94       815
# # # weighted avg       0.94      0.94      0.94       815




# # ------------------------------------------------------------------------------------------------------------------------

# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Train XGBoost model
# xgb_model = xgb.XGBClassifier(
#     objective="binary:logistic",
#     eval_metric="logloss"
#     )
# xgb_model.fit(X_train, y_train)

# print("XGBoost trained")

# # Predict with XGBoost
# xgb_pred = xgb_model.predict(X_test)
# xgb_acc = accuracy_score(y_test, xgb_pred)

# print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# # Classification Report
# print("\nXGBoost Classification Report:")
# print(classification_report(y_test, xgb_pred))

# # Confusion Matrix
# print("\nXGBoost Confusion Matrix:")
# xgb_cm = confusion_matrix(y_test, xgb_pred)
# sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues")
# plt.title("XGBoost Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# # XGBoost trained
# # XGBoost Accuracy: 0.9239

# # XGBoost Classification Report:
# #               precision    recall  f1-score   support

# #            0       0.93      0.93      0.93       446
# #            1       0.92      0.91      0.92       369

# #     accuracy                           0.92       815
# #    macro avg       0.92      0.92      0.92       815
# # weighted avg       0.92      0.92      0.92       815

# # ------------------------------------------------------------------------------------------------------------------------

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Train Random Forest model
# rf_model = RandomForestClassifier(
#     n_estimators=100,  # Number of trees in the forest
#     max_depth=None,    # No depth limit (can be adjusted to prevent overfitting)
#     random_state=42,   # Ensures reproducibility
#     n_jobs=-1          # Use all available CPU cores for faster training
# )
# rf_model.fit(X_train, y_train)

# print("Random Forest trained")

# # Predict with Random Forest
# rf_pred = rf_model.predict(X_test)
# rf_acc = accuracy_score(y_test, rf_pred)

# print(f"Random Forest Accuracy: {rf_acc:.4f}")

# # Classification Report
# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, rf_pred))

# # Confusion Matrix
# print("\nRandom Forest Confusion Matrix:")
# rf_cm = confusion_matrix(y_test, rf_pred)
# sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Random Forest Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# # Random Forest trained
# # Random Forest Accuracy: 0.9117

# # Random Forest Classification Report:
# #               precision    recall  f1-score   support

# #            0       0.91      0.93      0.92       446
# #            1       0.91      0.89      0.90       369

# #     accuracy                           0.91       815
# #    macro avg       0.91      0.91      0.91       815
# # weighted avg       0.91      0.91      0.91       815

# # ------------------------------------------------------------------------------------------------------------------------