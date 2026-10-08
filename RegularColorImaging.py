import cv2
import numpy as np

def segment_human_traditional(image_path, bbox):
    """
    Segments a human from an image using the traditional GrabCut algorithm.
    
    :param image_path: Path to the input RGB image
    :param bbox: A tuple of (x, y, width, height) bounding the human
    """
    # 1. Load the image in BGR (OpenCV standard)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    # Create a mask initialized with zeros (same spatial dimensions as image)
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Allocate temporary arrays used internally by the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # 2. Run GrabCut using the user-defined bounding box
    # 5 iterations is usually a good balance between speed and accuracy
    cv2.grabCut(img, mask, bbox, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # 3. Modify the mask to separate background from foreground
    # GrabCut modifies the mask: 0 & 2 are background, 1 & 3 are foreground
    # We map 0 and 2 to 0 (Sure/Probable Background) and 1 and 3 to 1 (Sure/Probable Foreground)
    binary_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
    
    # 4. Clean up boundaries using Morphological Operations
    # Closing operation fills small holes inside the segmented human
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Apply mask to the original image
    segmented_foreground = img * cleaned_mask[:, :, np.newaxis]
    
    # Find contours to draw the final exact boundary
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_img = img.copy()
    cv2.drawContours(boundary_img, contours, -1, (0, 255, 0), 2) # Green boundary
    
    # Save results
    cv2.imwrite("human_mask.png", cleaned_mask * 255)
    cv2.imwrite("human_segmented.png", segmented_foreground)
    cv2.imwrite("human_boundary.png", boundary_img)
    print("Segmentation complete. Images saved.")

# Example Usage:
# Define a bounding box around the person: (x, y, width, height)
bounding_box = (50, 50, 400, 600) 
segment_human_traditional("person.jpg", bounding_box)
