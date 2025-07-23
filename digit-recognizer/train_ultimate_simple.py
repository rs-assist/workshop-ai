#!/usr/bin/env python3
"""
🚀 ULTIMATE SIMPLE DIGIT RECOGNITION MODEL
================================================================================
A simplified but highly effective model focusing on proven techniques:
- Deep CNN with proven architecture
- Advanced data augmentation
- Ensemble training with multiple models
- Optimized hyperparameters
================================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cv2
from improved_preprocessing import preprocess_image

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_ultimate_simple_model(input_shape=(28, 28, 1)):
    """
    Create an ultimate CNN model with proven architecture
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First block - Feature extraction
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Second block - More complex features
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Third block - High-level features
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile with optimal settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def create_augmented_dataset(x_train, y_train, batch_size=128):
    """
    Create augmented dataset with advanced augmentation
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    return datagen.flow(x_train, y_train, batch_size=batch_size)

def train_ultimate_simple_model():
    """
    Train the ultimate simple model with best practices
    """
    print("🚀 TRAINING ULTIMATE SIMPLE DIGIT RECOGNITION MODEL")
    print("=" * 80)
    print("🌟 OPTIMIZED FEATURES:")
    print("- Deep CNN with proven architecture")
    print("- Advanced data augmentation")
    print("- Ensemble training for robustness")
    print("- Optimized callbacks and training")
    print("=" * 80)
    
    # Load and preprocess data
    print("🌟 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"📊 Training data shape: {x_train.shape}")
    print(f"📊 Test data shape: {x_test.shape}")
    
    # Create model
    print("🚀 Creating ultimate simple CNN model...")
    model = create_ultimate_simple_model()
    model.summary()
    
    # Create data augmentation
    print("🎨 Setting up advanced data augmentation...")
    train_generator = create_augmented_dataset(x_train, y_train)
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/ultimate_simple_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("🏋️ Training ultimate simple model...")
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // 128,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("📊 Evaluating ultimate simple model...")
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    print(f"🏆 Top-3 Accuracy: {test_top3_accuracy:.4f}")
    
    # Save model
    model.save('models/ultimate_simple_model.keras')
    print("💾 Model saved as 'models/ultimate_simple_model.keras'")
    
    # Train ensemble models for even better performance
    print("\n🚀 Training ensemble models for ultimate performance...")
    ensemble_models = []
    
    for i in range(3):
        print(f"🏋️ Training ensemble model {i+1}/3...")
        ensemble_model = create_ultimate_simple_model()
        
        # Use different random seeds for diversity
        tf.random.set_seed(42 + i)
        np.random.seed(42 + i)
        
        ensemble_generator = create_augmented_dataset(x_train, y_train)
        ensemble_history = ensemble_model.fit(
            ensemble_generator,
            steps_per_epoch=len(x_train) // 128,
            epochs=25,  # Slightly fewer epochs for ensemble
            validation_data=(x_test, y_test),
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.3,
                    patience=3,
                    min_lr=1e-7,
                    verbose=0
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=6,
                    restore_best_weights=True,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        ensemble_models.append(ensemble_model)
        ensemble_model.save(f'models/ultimate_ensemble_{i+1}.keras')
        print(f"💾 Ensemble model {i+1} saved")
    
    # Test ensemble performance
    print("\n📊 Testing ensemble performance...")
    ensemble_predictions = []
    for i, ensemble_model in enumerate(ensemble_models):
        pred = ensemble_model.predict(x_test[:1000], verbose=0)
        ensemble_predictions.append(pred)
    
    # Average ensemble predictions
    avg_ensemble_pred = np.mean(ensemble_predictions, axis=0)
    ensemble_accuracy = np.mean(np.argmax(avg_ensemble_pred, axis=1) == np.argmax(y_test[:1000], axis=1))
    print(f"🏆 Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Plot training history
    plot_ultimate_training_history(history)
    
    return model, history, ensemble_models

def plot_ultimate_training_history(history):
    """
    Plot training history with beautiful visualizations
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('🎯 Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('📉 Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot top-3 accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history.history['top_3_accuracy'], label='Training Top-3', linewidth=2)
    plt.plot(history.history['val_top_3_accuracy'], label='Validation Top-3', linewidth=2)
    plt.title('🏆 Top-3 Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultimate_simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Training history plot saved as 'ultimate_simple_training_history.png'")

if __name__ == "__main__":
    try:
        model, history, ensemble_models = train_ultimate_simple_model()
        print("\n✅ Ultimate simple model training completed successfully!")
        print("🏆 Ready for ultimate digit recognition performance!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
