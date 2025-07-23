"""
Simple script to recognize a hand-drawn digit from an image file
Usage: python recognize_digit.py path/to/your/image.png
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_image, extract_digit_from_image
from model_utils import load_model, predict_digit

def recognize_digit_from_file(image_path, model_path='models/digit_recognition_model.h5'):
    """
    Recognize a digit from an image file
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
    
    Returns:
        predicted_digit, confidence
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first.")
        return None, None
    
    # Load and process the image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Make sure the file exists and is a valid image format (PNG, JPG, etc.)")
        return None, None
    
    print("Processing image...")
    
    # Extract digit region (helps remove background noise)
    digit_region = extract_digit_from_image(image)
    
    # Preprocess for model prediction
    preprocessed = preprocess_image(digit_region)
    
    # Make prediction
    predicted_digit, confidence = predict_digit(model, preprocessed)
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Extracted digit region
    plt.subplot(1, 3, 2)
    plt.imshow(digit_region, cmap='gray')
    plt.title('Extracted Digit Region')
    plt.axis('off')
    
    # Preprocessed for model
    plt.subplot(1, 3, 3)
    plt.imshow(preprocessed.reshape(28, 28), cmap='gray')
    plt.title(f'Preprocessed\n(28x28 for model)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/digit_recognition_result.png')
    plt.show()
    
    return predicted_digit, confidence

def main():
    """
    Main function for command line usage
    """
    if len(sys.argv) != 2:
        print("Usage: python recognize_digit.py <path_to_image>")
        print("Example: python recognize_digit.py test_images/my_digit.png")
        return
    
    image_path = sys.argv[1]
    
    print("=" * 50)
    print("HANDWRITTEN DIGIT RECOGNITION")
    print("=" * 50)
    
    predicted_digit, confidence = recognize_digit_from_file(image_path)
    
    if predicted_digit is not None:
        print("\n" + "=" * 50)
        print("RESULTS:")
        print(f"Predicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        print("=" * 50)
        
        if confidence > 0.9:
            print("🎉 High confidence prediction!")
        elif confidence > 0.7:
            print("✓ Good confidence prediction")
        else:
            print("⚠ Low confidence - image quality might be poor")

if __name__ == "__main__":
    main()
