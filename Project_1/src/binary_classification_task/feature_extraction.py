import cv2
import os
import numpy as np
from skimage.feature import hog
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
cv2.setLogLevel(0)  # Suppress OpenCV warnings

# Define paths
DATASET_PATH = "../../dataset"
CATEGORIES = ["with_mask", "without_mask"]
IMAGE_SIZE = (128, 128)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
OUTPUT_FOLDER = "testing_features"  # Folder to store the .npy files

def extract_hog_features(img_path):
    """ Extract HOG features from a color image by processing each channel separately """
    try:
        img = cv2.imread(img_path)  # Read in color
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")  # Raise an error for debugging

        img = cv2.resize(img, IMAGE_SIZE)  # Resize for consistency
        hog_features = []

        # Extract HOG features from each color channel
        for channel in cv2.split(img):
            features = hog(channel, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
            hog_features.extend(features)

        return np.array(hog_features)
    except Exception as e:
        # print(f"Error processing {img_path}: {e}")  # Print error message
        return None  # Skip this image

def process_image(args):
    """Wrapper function to process a single image"""
    try:
        img_path, label = args
        features = extract_hog_features(img_path)
        return features, label
    except Exception as e:
        # print(f"Failed to process {img_path}: {e}")
        return None, None  # Skip this image

def load_dataset_parallel():
    """ Load dataset and extract features in parallel """
    image_paths_labels = []
    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(DATASET_PATH, category)
        try:
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                image_paths_labels.append((img_path, label))
        except Exception as e:
            # print(f"Skipping problematic folder: {folder_path}")
            continue

    X, y = [], []

    with Pool(max(cpu_count() - 1, 1)) as pool:
        results = list(tqdm(pool.imap(process_image, image_paths_labels), total=len(image_paths_labels)))

    for features, label in results:
        if features is not None and label is not None:
            X.append(features)
            y.append(label)


    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_dataset_parallel()

    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save extracted features
    np.save(os.path.join(OUTPUT_FOLDER, "X.npy"), X)
    np.save(os.path.join(OUTPUT_FOLDER, "y.npy"), y)

    print(f"Feature extraction completed\nX is {X.shape}\ny is {y.shape}")
    print(f"Features saved to: {OUTPUT_FOLDER}")