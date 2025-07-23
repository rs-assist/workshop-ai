#!/usr/bin/env python3
"""
Real-time webcam recognition module
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

class RealTimeDigitRecognizer:
    def __init__(self):
        self.models = []
        self.model_names = []
        self.load_models()
        
    def load_models(self):
        """Load all available models"""
        model_files = [
            ('models/ultimate_final_model.h5', 'Ultimate Final'),
            ('models/ultimate_final_ensemble_1.h5', 'Ensemble 1'),
            ('models/ultimate_final_ensemble_2.h5', 'Ensemble 2'),
            ('models/ultimate_final_ensemble_3.h5', 'Ensemble 3'),
            ('models/improved_digit_recognition_model.h5', 'Improved')
        ]
        
        for model_path, name in model_files:
            if os.path.exists(model_path):
                try:
                    model = keras.models.load_model(model_path)
                    self.models.append(model)
                    self.model_names.append(name)
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"⚠️ Could not load {name}: {e}")
        
        print(f"📊 Total models loaded: {len(self.models)}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Invert if needed
        if np.mean(resized) > 127:
            resized = 255 - resized
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        return normalized.reshape(1, 28, 28, 1)
    
    def predict_ensemble(self, processed_frame):
        """Make ensemble prediction"""
        if not self.models:
            return None, 0.0
        
        predictions = []
        for model in self.models:
            pred = model.predict(processed_frame, verbose=0)
            predictions.append(pred[0])
        
        # Ensemble average
        ensemble_avg = np.mean(predictions, axis=0)
        predicted_digit = np.argmax(ensemble_avg)
        confidence = ensemble_avg[predicted_digit]
        
        return predicted_digit, confidence
    
    def draw_prediction_info(self, frame, prediction, confidence):
        """Draw prediction information on frame"""
        height, width = frame.shape[:2]
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Prediction text
        cv2.putText(frame, f"Prediction: {prediction}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Confidence text
        conf_color = (0, 255, 0) if confidence >= 0.75 else (0, 255, 255) if confidence >= 0.60 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        # Model count
        cv2.putText(frame, f"Models: {len(self.models)}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to reset",
            "Press 'p' to pause"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 200, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def start_recognition(self):
        """Start real-time recognition"""
        print("🚀 STARTING REAL-TIME DIGIT RECOGNITION")
        print("=" * 50)
        print("📹 Opening webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        print("✅ Webcam opened successfully!")
        print("\n🎮 CONTROLS:")
        print("   • Press 'q' to quit")
        print("   • Press 'r' to reset")
        print("   • Press 'p' to pause/unpause")
        print("\n🎯 Point a handwritten digit at the camera!")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to read frame")
                        break
                    
                    # Make prediction
                    processed_frame = self.preprocess_frame(frame)
                    prediction, confidence = self.predict_ensemble(processed_frame)
                    
                    # Draw prediction info
                    if prediction is not None:
                        self.draw_prediction_info(frame, prediction, confidence)
                    
                    # Show frame
                    cv2.imshow('Real-time Digit Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("🔄 Reset requested")
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"⏸️ {status}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📹 Webcam closed")
            print("✅ Real-time recognition ended")

def start_realtime_recognition():
    """Entry point for real-time recognition"""
    recognizer = RealTimeDigitRecognizer()
    if not recognizer.models:
        print("❌ No models available for real-time recognition!")
        print("Please train models first using: python main.py train")
        return
    
    recognizer.start_recognition()
