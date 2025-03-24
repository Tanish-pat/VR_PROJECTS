import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import glob
import os
import argparse
from tqdm import tqdm

def load_dataset(msfd_path):
    """
    Load images and masks from MSFD/1/ directory.

    Parameters
    ----------
    msfd_path : str
        Path to the MSFD dataset.

    Returns
    -------
    image_paths, mask_paths : tuple of 2 lists of str
        Paths to the images and masks.
    """
    face_crop_dir = os.path.join(msfd_path, "1", "face_crop")
    mask_dir = os.path.join(msfd_path, "1", "face_crop_segmentation")

    # Check if the required directories exist
    if not os.path.exists(face_crop_dir):
        raise FileNotFoundError(f"face_crop directory not found in {msfd_path}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"face_crop_segmentation directory not found in {msfd_path}")

    # Get all image and mask files
    image_files = sorted(glob.glob(os.path.join(face_crop_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

    # Find common files between the two directories
    common_files = []
    for img_path in image_files:
        mask_path = os.path.join(mask_dir, os.path.basename(img_path))
        if mask_path in mask_files:
            common_files.append((img_path, mask_path))

    # Check if there are any common files
    if not common_files:
        raise ValueError("No matching image-mask pairs found")

    # Return the paths as separate lists
    return zip(*common_files)



def create_base_mask(image_rgb):
    """Create an enhanced mask using multiple techniques for more robust road detection.

    The steps in this function are:

    1. Color-based masking (extended with more road colors)
    2. Edge detection with Canny
    3. Dominant color detection
    4. Region of interest (ROI) - focus on the lower part of the image
    5. Combine all techniques
    6. Morphological operations to clean up the mask
    7. Keep only the largest contour
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    height, width = image_rgb.shape[:2]
    
    # 1. Color-based masking (extended with more road colors)
    # Blue, white, black (original)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 50, 50])
    # Add gray (common for concrete roads)
    lower_gray = np.array([0, 0, 70])
    upper_gray = np.array([180, 30, 140])
    # Add brown/tan (for dirt/gravel roads)
    lower_brown = np.array([10, 30, 60])
    upper_brown = np.array([30, 120, 150])
    
    # Create and combine color masks
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    color_mask = cv2.bitwise_or(mask_blue, mask_white)
    color_mask = cv2.bitwise_or(color_mask, mask_black)
    color_mask = cv2.bitwise_or(color_mask, mask_gray)
    color_mask = cv2.bitwise_or(color_mask, mask_brown)
    
    # 2. Edge detection with Canny
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby lines
    edge_kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # 3. Dominant color detection
    # Reshape and convert to float
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    
    # Define criteria for K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Determine the dominant color cluster for the lower part of the image
    lower_region = image_rgb[int(height*0.6):, :].reshape(-1, 3).astype(np.float32)
    lower_labels = np.zeros(lower_region.shape[0], dtype=np.uint8)
    
    # Assign each pixel to the nearest cluster center
    for i in range(lower_region.shape[0]):
        dists = np.sqrt(np.sum((lower_region[i] - centers)**2, axis=1))
        nearest_cluster = np.argmin(dists)
        lower_labels[i] = nearest_cluster
    
    # Find the most common cluster in the lower region
    counts = np.bincount(lower_labels)
    dominant_cluster = np.argmax(counts)
    
    # Create a mask for the dominant cluster
    dominant_mask = np.zeros((height, width), dtype=np.uint8)
    reshaped_labels = labels.reshape((height, width))
    dominant_mask[reshaped_labels == dominant_cluster] = 255
    
    # 4. Region of interest (ROI) - focus on the lower part of the image
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[int(height*0.4):, :] = 255
    
    # 5. Combine all techniques
    # Weight and combine the different masks
    final_mask = cv2.bitwise_and(color_mask, roi_mask)
    edge_roi = cv2.bitwise_and(dilated_edges, roi_mask)
    dominant_roi = cv2.bitwise_and(dominant_mask, roi_mask)
    
    combined_mask = cv2.bitwise_or(final_mask, edge_roi)
    combined_mask = cv2.bitwise_or(combined_mask, dominant_roi)
    
    # 6. Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
    
    # 7. Keep only the largest contour
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take the largest contour if it's significant enough
        if cv2.contourArea(contours[0]) > 1000:  # Minimum area threshold
            final_mask = np.zeros_like(processed_mask)
            cv2.drawContours(final_mask, [contours[0]], -1, 255, cv2.FILLED)
            return final_mask
        
        # If there are multiple significant contours, consider taking more than one
        if len(contours) > 1 and cv2.contourArea(contours[1]) > 0.5 * cv2.contourArea(contours[0]):
            final_mask = np.zeros_like(processed_mask)
            cv2.drawContours(final_mask, contours[:2], -1, 255, cv2.FILLED)
            return final_mask
    
    return processed_mask


def refine_with_grabcut(image_bgr, base_mask):
    """
    Refine the mask using GrabCut algorithm with ROI restriction.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR format.
    base_mask : np.ndarray
        Base mask to refine.

    Returns
    -------
    refined_mask : np.ndarray
        Refined mask with ROI restriction.
    """
    if np.count_nonzero(base_mask) < 100:
        return base_mask.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Initialize GrabCut mask with base mask
    grabcut_mask = np.where(base_mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    
    # Perform GrabCut algorithm
    cv2.grabCut(image_bgr, grabcut_mask, None, bgd_model, fgd_model, 
               iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
    
    # Get the refined mask by combining the FGD and PR_FGD labels
    refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Add ROI mask to black out top 40% (same as base mask)
    height, width = image_bgr.shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[int(height*0.5):, :] = 255 
    refined_mask = cv2.bitwise_and(refined_mask, roi_mask)  # Apply ROI mask

    return refined_mask

def evaluate_mask(pred_mask, true_mask):
    """
    Calculate Intersection over Union score.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted mask.
    true_mask : np.ndarray
        True mask.

    Returns
    -------
    iou : float
        Intersection over Union score.
    """
    # Resize the true mask to the same shape as the predicted mask
    if pred_mask.shape != true_mask.shape:
        true_mask = cv2.resize(true_mask, (pred_mask.shape[1], pred_mask.shape[0]))

    # Calculate the IoU score
    return jaccard_score(true_mask.flatten() > 127, pred_mask.flatten() > 127)


def main():
    parser = argparse.ArgumentParser(description="Mask Segmentation Evaluation")
    parser.add_argument("--mode", choices=["sample", "all"], default="all",
                       help="Processing mode: 'sample' or 'all'")
    parser.add_argument("--num", type=int, default=50,
                       help="Number of samples to process in sample mode")
    parser.add_argument("--visualize", type=int, default=25,
                       help="Number of samples to visualize")
    args = parser.parse_args()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    try:
        image_paths, mask_paths = load_dataset("../../../dataset/MSFD/")
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.mode == "sample":
        indices = np.random.choice(len(image_paths), min(args.num, len(image_paths)), False)
        image_paths = [image_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]

    total_base_iou = 0.0
    total_grabcut_iou = 0.0
    base_ious = []
    grabcut_ious = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for idx, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths))):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Skipping invalid image: {os.path.basename(img_path)}")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if true_mask is None:
            print(f"Skipping invalid mask: {os.path.basename(mask_path)}")
            continue
            
        true_mask = (true_mask > 128).astype(np.uint8) * 255

        base_mask = create_base_mask(image_rgb)
        base_iou = calculate_iou(base_mask > 0, true_mask > 0)
        total_base_iou += base_iou
        base_ious.append(base_iou)
        
        grabcut_mask = refine_with_grabcut(image_bgr, base_mask)
        grabcut_iou = calculate_iou(grabcut_mask > 0, true_mask > 0)
        total_grabcut_iou += grabcut_iou
        grabcut_ious.append(grabcut_iou)

        print(f"\nImage {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        print(f"  Base Mask IoU: {base_iou:.4f}")
        print(f"  GrabCut Mask IoU: {grabcut_iou:.4f}")
        
        if idx < args.visualize:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 4, 1)
            plt.imshow(image_rgb)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(base_mask, cmap='gray')
            plt.title(f'Base Mask\nIoU: {base_iou:.3f}')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(grabcut_mask, cmap='gray')
            plt.title(f'GrabCut Mask\nIoU: {grabcut_iou:.3f}')
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.imshow(true_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"mask_comparison_{idx}.png"))
            plt.close()

    n_samples = len(image_paths)
    avg_base = total_base_iou / n_samples
    avg_grabcut = total_grabcut_iou / n_samples
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total images processed: {n_samples}")
    print(f"Base Mask:        IoU = {avg_base:.4f}")
    print(f"GrabCut Mask:     IoU = {avg_grabcut:.4f}")
    
    improvement = (avg_grabcut - avg_base) / avg_base * 100
    print(f"GrabCut improvement: {improvement:.2f}%")
    
    methods = ['Base Mask', 'GrabCut Mask']
    iou_avgs = [avg_base, avg_grabcut]
    
    plt.bar(methods, iou_avgs)
    plt.ylabel('IoU Score')
    plt.title('Comparison of Base Mask and GrabCut Mask')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "iou_comparison.png"))
    plt.close()
    
    plt.hist(base_ious, bins=50, alpha=0.5, label='Base Mask')
    plt.hist(grabcut_ious, bins=50, alpha=0.5, label='GrabCut Mask')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "iou_distribution.png"))
    plt.close()
    
    with open(os.path.join(results_dir, "evaluation_summary.txt"), "w") as f:
        f.write("FINAL EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Total images processed: {n_samples}\n")
        f.write(f"Base Mask:        IoU = {avg_base:.4f}\n")
        f.write(f"GrabCut Mask:     IoU = {avg_grabcut:.4f}\n")
        f.write(f"GrabCut improvement: {improvement:.2f}%\n")

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0

if __name__ == "__main__":
    main()