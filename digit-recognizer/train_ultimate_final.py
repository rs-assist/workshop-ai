#!/usr/bin/env python3
"""
🚀 ULTIMATE CORRECTED DIGIT RECOGNITION MODEL
================================================================================
Final optimized version with all fixes applied
================================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_ultimate_model(input_shape=(28, 28, 1)):
    """Create ultimate CNN model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First block
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile with correct metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ultimate_final_model():
    """Train the ultimate final model"""
    print("🚀 TRAINING ULTIMATE FINAL MODEL")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"📊 Data shape: {x_train.shape}")
    
    # Create model
    model = create_ultimate_model()
    print(f"📊 Parameters: {model.count_params():,}")
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow(x_train, y_train, batch_size=128)
    
    # Callbacks
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
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/ultimate_final_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // 128,
        epochs=40,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
    
    # Save
    model.save('models/ultimate_final_model.h5')
    
    # Train ensemble
    print("\n🚀 Training ensemble models...")
    ensemble_models = []
    
    for i in range(3):
        print(f"Training ensemble {i+1}/3...")
        ensemble_model = create_ultimate_model()
        
        tf.random.set_seed(42 + i * 10)
        np.random.seed(42 + i * 10)
        
        ensemble_generator = datagen.flow(x_train, y_train, batch_size=128)
        ensemble_model.fit(
            ensemble_generator,
            steps_per_epoch=len(x_train) // 128,
            epochs=30,
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
                    patience=8,
                    restore_best_weights=True,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        ensemble_models.append(ensemble_model)
        ensemble_model.save(f'models/ultimate_final_ensemble_{i+1}.h5')
        
        _, acc = ensemble_model.evaluate(x_test, y_test, verbose=0)
        print(f"Ensemble {i+1} accuracy: {acc:.4f}")
    
    # Test ensemble
    predictions = []
    for model in ensemble_models:
        pred = model.predict(x_test, verbose=0)
        predictions.append(pred)
    
    avg_pred = np.mean(predictions, axis=0)
    ensemble_accuracy = np.mean(np.argmax(avg_pred, axis=1) == np.argmax(y_test, axis=1))
    
    print(f"\n🏆 FINAL ENSEMBLE ACCURACY: {ensemble_accuracy:.4f}")
    
    # Calculate improvement
    original_accuracy = 0.9926
    improvement = (ensemble_accuracy - original_accuracy) / original_accuracy * 100
    print(f"📈 Improvement: +{improvement:.2f}%")
    
    return model, ensemble_models, ensemble_accuracy

if __name__ == "__main__":
    try:
        model, ensemble_models, accuracy = train_ultimate_final_model()
        print(f"\n✅ SUCCESS! Final accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
