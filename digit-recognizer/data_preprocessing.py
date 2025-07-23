"""
Data preprocessing utilities for digit recognition
"""
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_image(image, target_size=(28, 28)):
    """
    Preprocess an image for digit recognition
    
    Args:
        image: Input image (BGR or grayscale)
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed image ready for model prediction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize image
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    preprocessed = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return preprocessed

def preprocess_mnist_data(x_train, y_train, x_test, y_test, num_classes=10):
    """
    Preprocess MNIST dataset
    
    Args:
        x_train, y_train, x_test, y_test: MNIST data
        num_classes: Number of classes (digits 0-9)
    
    Returns:
        Preprocessed training and testing data
    """
    # Reshape data to add channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

def extract_digit_from_image(image):
    """
    Extract digit region from an image using contour detection
    
    Args:
        image: Input image containing a digit
    
    Returns:
        Extracted digit region
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract digit region with some padding
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(gray.shape[1], x + w + padding)
        y_end = min(gray.shape[0], y + h + padding)
        
        digit_region = gray[y_start:y_end, x_start:x_end]
        
        return digit_region
    
    return gray
