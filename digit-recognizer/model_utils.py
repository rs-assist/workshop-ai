"""
Model utilities for digit recognition
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a Convolutional Neural Network for digit recognition
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes (digits 0-9)
    
    Returns:
        Compiled CNN model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, model_path):
    """
    Save the trained model
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model
    """
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None

def predict_digit(model, image):
    """
    Predict digit from preprocessed image
    
    Args:
        model: Trained model
        image: Preprocessed image
    
    Returns:
        Predicted digit and confidence
    """
    predictions = model.predict(image, verbose=0)
    predicted_digit = predictions.argmax()
    confidence = predictions.max()
    
    return predicted_digit, confidence
