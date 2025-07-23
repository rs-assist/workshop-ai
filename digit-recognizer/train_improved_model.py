"""
Improved training script with data augmentation and better preprocessing
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_mnist_data
from model_utils import create_cnn_model, save_model

def create_improved_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create an improved CNN model with better architecture
    """
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a better optimizer with learning rate scheduling
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_augmentation():
    """
    Create data augmentation generator for better training
    """
    datagen = ImageDataGenerator(
        rotation_range=10,          # Rotate images by up to 10 degrees
        width_shift_range=0.1,      # Shift images horizontally
        height_shift_range=0.1,     # Shift images vertically
        shear_range=0.1,           # Shear transformation
        zoom_range=0.1,            # Zoom in/out
        fill_mode='nearest'        # Fill pixels after transformation
    )
    return datagen

def train_improved_model():
    """
    Train an improved model with data augmentation
    """
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    x_train, y_train, x_test, y_test = preprocess_mnist_data(x_train, y_train, x_test, y_test)
    
    # Create improved model
    print("Creating improved CNN model...")
    model = create_improved_cnn_model()
    
    print("Model architecture:")
    model.summary()
    
    # Create data augmentation
    print("Setting up data augmentation...")
    datagen = create_data_augmentation()
    datagen.fit(x_train)
    
    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=0.0001,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train with data augmentation
    print("Training improved model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("Evaluating improved model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Improved Model Test Accuracy: {test_accuracy:.4f}")
    
    # Save improved model
    model_path = 'models/improved_digit_recognition_model.h5'
    save_model(model, model_path)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Plot training history for improved model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Improved Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Improved Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/improved_training_history.png')
    plt.show()

if __name__ == "__main__":
    print("🚀 TRAINING IMPROVED DIGIT RECOGNITION MODEL")
    print("="*60)
    print("Improvements:")
    print("- Better CNN architecture with BatchNormalization")
    print("- Data augmentation (rotation, shifting, zoom)")
    print("- Learning rate scheduling")
    print("- Early stopping")
    print("- More layers and dropout for better generalization")
    print("="*60)
    
    model, history = train_improved_model()
    print("\n✅ Improved model training completed!")
    print("🔄 You can now test it with: python predict_digit.py test_images/your_image.png")
    print("📁 Model saved as: models/improved_digit_recognition_model.h5")
