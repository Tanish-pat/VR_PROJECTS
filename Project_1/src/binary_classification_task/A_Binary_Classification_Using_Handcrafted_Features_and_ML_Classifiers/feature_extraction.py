import os
import numpy as np
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
cv2.setLogLevel(0)

# Define paths
DATASET_PATH = "../../../dataset"
CATEGORIES = ["with_mask", "without_mask"]
IMAGE_SIZE = (128, 128)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
FEATURES_FOLDER = "enhanced_features"
EPOCHS = 10
BATCH_SIZE = 32

# Create output folder if not exists
os.makedirs(FEATURES_FOLDER, exist_ok=True)

def extract_features(img_path):
    """ Extract HOG, Color Histogram, LBP features and Edge Histogram from an image """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, IMAGE_SIZE)

        # HOG Features
        hog_features = [hog(channel, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
                        for channel in cv2.split(img)]
        hog_features = np.hstack(hog_features)

        # Color Histogram Features (using HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_features = np.concatenate([cv2.calcHist([hsv], [i], None, [8], [0, 256]).flatten() for i in range(3)])

        # Local Binary Patterns (LBP)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize

        # Edge Histogram
        edges = cv2.Canny(gray, 100, 200)
        edge_hist, _ = np.histogram(edges, bins=8, range=(0, 256))
        edge_hist = edge_hist.astype("float")
        edge_hist /= (edge_hist.sum() + 1e-7)  # Normalize

        # Concatenate all features
        final_features = np.hstack([hog_features, hist_features, lbp_hist, edge_hist])
        return final_features
    except:
        return None

def process_image(args):
    """ Process single image for multiprocessing """
    img_path, label = args
    features = extract_features(img_path)
    return (features, label) if features is not None else (None, None)

def load_dataset_parallel():
    """ Load dataset and extract features in parallel """
    image_paths_labels = [(os.path.join(DATASET_PATH, cat, f), label)
                            for label, cat in enumerate(CATEGORIES)
                            for f in os.listdir(os.path.join(DATASET_PATH, cat))]

    X_hog, y_hog = [], []
    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap(process_image, image_paths_labels), total=len(image_paths_labels)))

    for features, label in results:
        if features is not None:
            X_hog.append(features)
            y_hog.append(label)

    return np.array(X_hog), np.array(y_hog)

if __name__ == '__main__':  # Ensure this part only runs in the main process
    # Run feature extraction only if data is missing
    print("Extracting features...")
    X_hog, y_hog = load_dataset_parallel()
    np.save(os.path.join(FEATURES_FOLDER, "X.npy"), X_hog)
    np.save(os.path.join(FEATURES_FOLDER, "y.npy"), y_hog)
    print(f"Feature extraction completed. Saved {X_hog.shape}.")
    print(f"Shape of X_hog: {X_hog.shape}")
    print(f"Shape of y_hog: {y_hog.shape}")