"""
Test digit recognition on individual image files
"""
import cv2
import os
from data_preprocessing import preprocess_image, extract_digit_from_image
from model_utils import load_model, predict_digit

def test_image_file(model, image_path):
    """
    Test digit recognition on a single image file
    
    Args:
        model: Trained model
        image_path: Path to image file
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    
    # Extract digit region
    digit_region = extract_digit_from_image(image)
    
    # Preprocess for model
    preprocessed = preprocess_image(digit_region)
    
    # Make prediction
    predicted_digit, confidence = predict_digit(model, preprocessed)
    
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.3f}")
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Extracted Digit', digit_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main function to test images in test_images directory
    """
    # Load the trained model
    model_path = 'models/digit_recognition_model.h5'
    print(f"Loading model from {model_path}...")
    
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first by running train_model.py")
        return
    
    # Test images in test_images directory
    test_dir = 'test_images'
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        print("Please add some digit images to the test_images/ directory")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No image files found in test_images/ directory")
        print("Please add some digit images to test")
        return
    
    print(f"Found {len(image_files)} image(s) to test")
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        test_image_file(model, image_path)
        
        # Ask user to continue
        response = input("Press Enter to continue to next image, or 'q' to quit: ")
        if response.lower() == 'q':
            break
    
    print("Testing completed!")

if __name__ == "__main__":
    main()
