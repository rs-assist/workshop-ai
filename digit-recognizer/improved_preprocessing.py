"""
Improved preprocessing for better digit recognition on real images
"""
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def improved_preprocess_image(image, target_size=(28, 28), show_steps=False):
    """
    Improved preprocessing for better recognition of hand-drawn digits
    
    Args:
        image: Input image (BGR or grayscale)
        target_size: Target size for the image
        show_steps: Whether to show preprocessing steps
    
    Returns:
        Preprocessed image ready for model prediction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold for better binarization
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove small noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours and get the largest one
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle with padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = max(10, int(min(w, h) * 0.1))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        # Extract digit region
        digit_region = cleaned[y:y+h, x:x+w]
    else:
        digit_region = cleaned
    
    # Make the image square by adding padding
    h, w = digit_region.shape
    max_dim = max(h, w)
    
    # Create square canvas
    square_image = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    # Center the digit in the square canvas
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = digit_region
    
    # Resize to target size
    resized = cv2.resize(square_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Apply slight Gaussian blur to match MNIST style
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    
    # Reshape for model input
    preprocessed = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    if show_steps:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 3))
        
        plt.subplot(1, 5, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original Grayscale')
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Threshold')
        plt.axis('off')
        
        plt.subplot(1, 5, 3)
        plt.imshow(cleaned, cmap='gray')
        plt.title('Noise Removed')
        plt.axis('off')
        
        plt.subplot(1, 5, 4)
        plt.imshow(square_image, cmap='gray')
        plt.title('Squared & Centered')
        plt.axis('off')
        
        plt.subplot(1, 5, 5)
        plt.imshow(resized, cmap='gray')
        plt.title('Final 28x28')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return preprocessed

def extract_digit_improved(image):
    """
    Improved digit extraction with better contour detection
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area (remove very small ones)
        min_area = 100  # Minimum area for a valid digit
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add smart padding based on digit size
            padding = max(5, int(min(w, h) * 0.2))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            # Extract region
            digit_region = gray[y:y+h, x:x+w]
            return digit_region
    
    # If no good contours found, return original
    return gray
