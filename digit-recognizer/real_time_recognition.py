"""
Real-time digit recognition using camera input
"""
import cv2
import numpy as np
from data_preprocessing import preprocess_image, extract_digit_from_image
from model_utils import load_model, predict_digit

def setup_camera():
    """
    Setup camera for real-time capture
    
    Returns:
        Camera object
    """
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def draw_prediction_overlay(frame, digit, confidence, x, y, w, h):
    """
    Draw prediction overlay on frame
    
    Args:
        frame: Input frame
        digit: Predicted digit
        confidence: Prediction confidence
        x, y, w, h: Bounding box coordinates
    """
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw prediction text
    text = f"Digit: {digit} ({confidence:.2f})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)

def main():
    """
    Main real-time recognition function
    """
    # Load the trained model
    model_path = 'models/digit_recognition_model.h5'
    print(f"Loading model from {model_path}...")
    
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first by running train_model.py")
        return
    
    # Setup camera
    print("Setting up camera...")
    cap = setup_camera()
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Real-time digit recognition started!")
    print("Instructions:")
    print("- Hold a written digit in front of the camera")
    print("- Press 'c' to capture and recognize a digit")
    print("- Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display instructions on frame
        cv2.putText(frame, "Press 'c' to capture digit, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Digit Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture and process the current frame
            print("Capturing digit...")
            
            # Extract digit region
            digit_region = extract_digit_from_image(frame)
            
            if digit_region is not None:
                # Preprocess for model
                preprocessed = preprocess_image(digit_region)
                
                # Make prediction
                predicted_digit, confidence = predict_digit(model, preprocessed)
                
                print(f"Predicted digit: {predicted_digit} (confidence: {confidence:.3f})")
                
                # Show extracted digit region
                cv2.imshow('Extracted Digit', digit_region)
                
                # Optional: Save the captured digit
                cv2.imwrite(f'test_images/captured_digit_{predicted_digit}.png', digit_region)
            else:
                print("No digit detected in the frame")
        
        elif key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time recognition stopped.")

if __name__ == "__main__":
    main()
