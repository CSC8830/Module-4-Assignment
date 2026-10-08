import cv2
import numpy as np

def segment_human_thermal(image_path):
    """
    Segments a human from a thermal image using adaptive intensity thresholding.
    
    :param image_path: Path to the input thermal image (Grayscale or False-color)
    """
    # 1. Load image 
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Thermal image could not be loaded.")
        return
    
    # 2. Convert to grayscale if it is saved in a false-color format (e.g., Ironbow or Jet)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Gaussian Blur to eliminate high-frequency thermal sensor noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 4. Otsu's Thresholding (Automatically calculates the optimal heat threshold)
    # Assumes the human body is hotter (brighter) than the environment
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphological Operations to join separated hot regions (e.g., cold clothing blocking core heat)
    # We use a vertical rectangle kernel because humans are standing/vertical shapes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small environment heat artifacts (e.g., electronics, rocks heated by sun)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 6. Isolate the human boundary using Contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundary_img = img.copy()
    
    # Filter contours by minimum area to ignore leftover thermal noise
    min_area = 500  
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Draw exact boundary around the heat signature
            cv2.drawContours(boundary_img, [contour], -1, (0, 0, 255), 2)  # Red boundary
            
    # Save the specialized outputs
    cv2.imwrite("thermal_mask.png", cleaned_mask)
    cv2.imwrite("thermal_human_boundary.png", boundary_img)
    print("Thermal segmentation processing complete.")
# Run example (replace with your thermal file)
segment_human_thermal("person.jpg")
