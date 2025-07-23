#!/usr/bin/env python3
"""
Single image prediction module
"""

import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Additional TensorFlow warnings suppression
tf.get_logger().setLevel('ERROR')

def load_all_models():
    """Load all available trained models"""
    models = []
    model_names = []
    
    model_files = [
        ('models/ultimate_final_model.h5', 'Ultimate Final'),
        ('models/ultimate_final_ensemble_1.h5', 'Final Ensemble 1'),
        ('models/ultimate_final_ensemble_2.h5', 'Final Ensemble 2'),
        ('models/ultimate_final_ensemble_3.h5', 'Final Ensemble 3'),
    ]
    
    for model_path, name in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path, compile=False)
                # Compile the model to avoid warnings
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                models.append(model)
                model_names.append(name)
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Could not load {name}: {e}")
    
    return models, model_names

def preprocess_image_simple(image_path):
    """Simple but effective image preprocessing"""
    try:
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Invert if needed (make digits white on black background)
        if np.mean(image) > 127:
            image = 255 - image
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_digit(models, model_names, image_path):
    """Predict digit using ensemble of models"""
    print(f"\nAnalyzing: {image_path}")
    print("-" * 40)
    
    # Preprocess image
    processed_image = preprocess_image_simple(image_path)
    if processed_image is None:
        print("ERROR: Could not process image")
        return None
    
    # Predict with all models
    predictions = []
    for model, name in zip(models, model_names):
        pred = model.predict(processed_image.reshape(1, 28, 28, 1), verbose=0)
        predicted_digit = np.argmax(pred[0])
        confidence = pred[0][predicted_digit]
        predictions.append(pred[0])
        print(f"{name:20} -> {predicted_digit} ({confidence:.1%})")
    
    # Ensemble prediction
    if len(predictions) > 1:
        ensemble_avg = np.mean(predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_avg)
        ensemble_conf = ensemble_avg[ensemble_pred]
        
        print("-" * 40)
        print(f"ENSEMBLE RESULT -> {ensemble_pred} ({ensemble_conf:.1%})")
        
        return ensemble_pred, ensemble_conf
    else:
        return predicted_digit, confidence

def predict_single_image(image_path):
    """Main function to predict a single image"""
    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        return
    
    print("🚀 DIGIT RECOGNITION SYSTEM")
    print("=" * 40)
    
    # Load models
    models, model_names = load_all_models()
    if not models:
        print("ERROR: No models found!")
        return
    
    print(f"Loaded {len(models)} models")
    
    # Make prediction
    result = predict_digit(models, model_names, image_path)
    if result:
        pred, conf = result
        print(f"\nFINAL PREDICTION: {pred}")
        print(f"CONFIDENCE: {conf:.1%}")
        
        # Confidence assessment
        if conf >= 0.95:
            print("Confidence Level: VERY HIGH")
        elif conf >= 0.80:
            print("Confidence Level: HIGH")
        elif conf >= 0.65:
            print("Confidence Level: MEDIUM")
        else:
            print("Confidence Level: LOW")
            
        return pred, conf
    
    return None
