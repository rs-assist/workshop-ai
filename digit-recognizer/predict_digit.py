"""
Simple script to test hand-drawn digit images
Usage: python predict_digit.py path/to/your/image.png
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_image, extract_digit_from_image
from model_utils import load_model, predict_digit

def predict_single_image(image_path):
    """
    Predict digit from a single image file
    
    Args:
        image_path: Path to the image file
    """
    # Load the trained model
    model_path = 'models/digit_recognition_model.h5'
    print(f"Loading model from {model_path}...")
    
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Load and process the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        print("Supported formats: .png, .jpg, .jpeg, .bmp")
        return
    
    # Show original image
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Extract digit region
    digit_region = extract_digit_from_image(image)
    
    # Show extracted digit
    plt.subplot(1, 3, 2)
    plt.imshow(digit_region, cmap='gray')
    plt.title('Extracted Digit Region')
    plt.axis('off')
    
    # Preprocess for model
    preprocessed = preprocess_image(digit_region)
    
    # Show preprocessed image (28x28)
    plt.subplot(1, 3, 3)
    plt.imshow(preprocessed.reshape(28, 28), cmap='gray')
    plt.title('Preprocessed (28x28)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Make prediction
    predicted_digit, confidence = predict_digit(model, preprocessed)
    
    # Display results
    print("\n" + "="*50)
    print(f"PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("="*50)
    
    if confidence > 0.9:
        print("🟢 High confidence prediction!")
    elif confidence > 0.7:
        print("🟡 Medium confidence prediction")
    else:
        print("🔴 Low confidence prediction - image might be unclear")

def main():
    """
    Main function
    """
    if len(sys.argv) != 2:
        print("Usage: python predict_digit.py <path_to_image>")
        print("Example: python predict_digit.py test_images/my_digit.png")
        return
    
    image_path = sys.argv[1]
    predict_single_image(image_path)

if __name__ == "__main__":
    main()
