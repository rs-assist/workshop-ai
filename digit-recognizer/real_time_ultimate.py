#!/usr/bin/env python3
"""
🎥 REAL-TIME ULTIMATE DIGIT RECOGNITION
Live webcam digit recognition with state-of-the-art models!

Features:
- Real-time webcam input
- Live preprocessing visualization
- Multiple model ensemble predictions
- Confidence tracking over time
- Gesture-based controls
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from improved_preprocessing import improved_preprocess_image

class RealTimeDigitRecognizer:
    def __init__(self):
        self.models = self.load_models()
        self.confidence_history = deque(maxlen=30)  # Last 30 predictions
        self.prediction_history = deque(maxlen=10)  # Last 10 predictions
        self.stable_prediction = None
        self.stable_confidence = 0.0
        self.last_prediction_time = time.time()
        
    def load_models(self):
        """Load all available models"""
        models = {}
        
        # Load ultimate model
        ultimate_path = 'models/ultimate_digit_recognition_model.h5'
        if os.path.exists(ultimate_path):
            print("🚀 Loading ultimate model...")
            models['ultimate'] = keras.models.load_model(ultimate_path)
        
        # Load improved model
        improved_path = 'models/improved_digit_recognition_model.h5'
        if os.path.exists(improved_path):
            print("🔧 Loading improved model...")
            models['improved'] = keras.models.load_model(improved_path)
        
        # Load ensemble models
        ensemble_models = []
        for i in range(1, 4):
            ensemble_path = f'models/ensemble_model_{i}.h5'
            if os.path.exists(ensemble_path):
                print(f"🤖 Loading ensemble model {i}...")
                ensemble_models.append(keras.models.load_model(ensemble_path))
        
        if ensemble_models:
            models['ensemble'] = ensemble_models
        
        return models
    
    def extract_digit_region(self, frame):
        """Extract digit region from frame using advanced techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter by size (digits should be reasonably sized)
        if w < 30 or h < 30 or w > 200 or h > 200:
            return None, None, None
        
        # Extract digit region
        digit_region = gray[y:y+h, x:x+w]
        
        return digit_region, (x, y, w, h), thresh
    
    def preprocess_for_prediction(self, digit_region):
        """Preprocess digit region for model prediction"""
        if digit_region is None:
            return None
        
        # Create a 3-channel image for improved_preprocess_image
        digit_bgr = cv2.cvtColor(digit_region, cv2.COLOR_GRAY2BGR)
        
        # Use improved preprocessing
        preprocessed = improved_preprocess_image(digit_bgr)
        
        return preprocessed
    
    def ensemble_predict(self, preprocessed_image):
        """Make ensemble prediction"""
        if preprocessed_image is None:
            return None, 0.0, {}
        
        all_predictions = []
        model_results = {}
        
        # Predict with ultimate model
        if 'ultimate' in self.models:
            pred = self.models['ultimate'].predict(preprocessed_image, verbose=0)
            model_results['ultimate'] = {
                'prediction': np.argmax(pred[0]),
                'confidence': float(np.max(pred[0]))
            }
            all_predictions.append(pred[0])
        
        # Predict with improved model
        if 'improved' in self.models:
            pred = self.models['improved'].predict(preprocessed_image, verbose=0)
            model_results['improved'] = {
                'prediction': np.argmax(pred[0]),
                'confidence': float(np.max(pred[0]))
            }
            all_predictions.append(pred[0])
        
        # Predict with ensemble models
        if 'ensemble' in self.models:
            for i, model in enumerate(self.models['ensemble']):
                pred = model.predict(preprocessed_image, verbose=0)
                model_results[f'ensemble_{i+1}'] = {
                    'prediction': np.argmax(pred[0]),
                    'confidence': float(np.max(pred[0]))
                }
                all_predictions.append(pred[0])
        
        if not all_predictions:
            return None, 0.0, {}
        
        # Ensemble prediction
        ensemble_pred = np.mean(all_predictions, axis=0)
        final_prediction = np.argmax(ensemble_pred)
        final_confidence = float(np.max(ensemble_pred))
        
        return final_prediction, final_confidence, model_results
    
    def update_stable_prediction(self, prediction, confidence):
        """Update stable prediction based on history"""
        self.prediction_history.append((prediction, confidence))
        self.confidence_history.append(confidence)
        
        # Only update stable prediction if we have enough history
        if len(self.prediction_history) >= 5:
            # Get most common prediction in recent history
            recent_predictions = [p[0] for p in list(self.prediction_history)[-5:]]
            prediction_counts = {}
            for p in recent_predictions:
                prediction_counts[p] = prediction_counts.get(p, 0) + 1
            
            # Get most frequent prediction
            most_frequent = max(prediction_counts.items(), key=lambda x: x[1])
            
            # Update stable prediction if it's consistent and confident
            if most_frequent[1] >= 3:  # Appears at least 3 times in last 5
                recent_confidences = [p[1] for p in list(self.prediction_history)[-5:] 
                                    if p[0] == most_frequent[0]]
                avg_confidence = np.mean(recent_confidences)
                
                if avg_confidence > 0.7:  # High confidence threshold
                    self.stable_prediction = most_frequent[0]
                    self.stable_confidence = avg_confidence
    
    def draw_predictions(self, frame, bbox, prediction, confidence, model_results):
        """Draw predictions and information on frame"""
        height, width = frame.shape[:2]
        
        # Draw bounding box if digit detected
        if bbox is not None:
            x, y, w, h = bbox
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Create info panel
        panel_height = 200
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Draw prediction info
        if prediction is not None:
            # Main prediction
            cv2.putText(panel, f"Prediction: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(panel, f"Confidence: {confidence:.1%}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stable prediction
            if self.stable_prediction is not None:
                cv2.putText(panel, f"Stable: {self.stable_prediction} ({self.stable_confidence:.1%})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Model agreement
            y_offset = 120
            for name, result in model_results.items():
                color = (0, 255, 0) if result['prediction'] == prediction else (0, 0, 255)
                cv2.putText(panel, f"{name}: {result['prediction']} ({result['confidence']:.1%})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        # Draw confidence history graph
        if len(self.confidence_history) > 1:
            graph_x = width - 200
            graph_y = 10
            graph_width = 180
            graph_height = 80
            
            # Background
            cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                         (50, 50, 50), -1)
            
            # Plot confidence history
            confidences = list(self.confidence_history)
            for i in range(1, len(confidences)):
                x1 = graph_x + int((i-1) * graph_width / len(confidences))
                y1 = graph_y + graph_height - int(confidences[i-1] * graph_height)
                x2 = graph_x + int(i * graph_width / len(confidences))
                y2 = graph_y + graph_height - int(confidences[i] * graph_height)
                cv2.line(panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            cv2.putText(panel, "Confidence History", (graph_x, graph_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(panel, "Press 'q' to quit, 's' to save screenshot", (10, panel_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        return combined
    
    def run(self):
        """Run real-time digit recognition"""
        print("🎥 Starting real-time digit recognition...")
        print("📋 Instructions:")
        print("   - Show a digit to the camera")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'r' to reset stable prediction")
        
        if not self.models:
            print("❌ No models loaded! Please train models first.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🚀 Camera started! Show digits to the camera...")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract digit region
            digit_region, bbox, thresh = self.extract_digit_region(frame)
            
            # Preprocess for prediction
            preprocessed = self.preprocess_for_prediction(digit_region)
            
            # Make prediction
            prediction = None
            confidence = 0.0
            model_results = {}
            
            if preprocessed is not None:
                prediction, confidence, model_results = self.ensemble_predict(preprocessed)
                
                if prediction is not None:
                    self.update_stable_prediction(prediction, confidence)
            
            # Draw results
            display_frame = self.draw_predictions(frame, bbox, prediction, confidence, model_results)
            
            # Show frame
            cv2.imshow('Ultimate Digit Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f'screenshot_{screenshot_count:03d}.png'
                cv2.imwrite(screenshot_name, display_frame)
                print(f"Screenshot saved: {screenshot_name}")
                screenshot_count += 1
            elif key == ord('r'):
                self.stable_prediction = None
                self.stable_confidence = 0.0
                self.prediction_history.clear()
                self.confidence_history.clear()
                print("Reset stable prediction")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time recognition ended.")

def main():
    """Main function"""
    print("ULTIMATE REAL-TIME DIGIT RECOGNITION")
    print("=" * 50)
    
    recognizer = RealTimeDigitRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()
