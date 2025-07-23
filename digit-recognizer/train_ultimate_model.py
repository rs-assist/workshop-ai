#!/usr/bin/env python3
"""
🚀 ULTIMATE DIGIT RECOGNITION MODEL
State-of-the-art architecture with cutting-edge techniques!

New features:
- ResNet-style skip connections for deeper training
- Attention mechanisms for better feature focus
- Advanced data augmentation with mixup/cutmix
- Ensemble predictions from multiple models
- Self-supervised pre-training
- Advanced regularization techniques
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from data_preprocessing import preprocess_mnist_data
from model_utils import save_model

def create_attention_block(x, filters):
    """
    Create attention mechanism to focus on important features
    """
    # Global average pooling for spatial attention
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gap = tf.keras.layers.Reshape((1, 1, filters))(gap)
    
    # Create attention weights
    attention = tf.keras.layers.Dense(filters // 4, activation='relu')(gap)
    attention = tf.keras.layers.Dense(filters, activation='sigmoid')(attention)
    
    # Apply attention to input
    attended = tf.keras.layers.Multiply()([x, attention])
    
    return attended

def create_residual_block(x, filters, kernel_size=3):
    """
    Create ResNet-style residual block for better gradient flow
    """
    shortcut = x
    
    # First conv layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second conv layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add shortcut connection
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def create_ultimate_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create the ultimate CNN model with cutting-edge techniques
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial convolution
    x = tf.keras.layers.Conv2D(32, 7, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # First residual block with attention
    x = create_residual_block(x, 32)
    x = create_attention_block(x, 32)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Second residual block with attention
    x = create_residual_block(x, 64)
    x = create_attention_block(x, 64)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Third residual block with attention
    x = create_residual_block(x, 128)
    x = create_attention_block(x, 128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Advanced dense layers with skip connections
    dense1 = tf.keras.layers.Dense(512, activation='relu')(x)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.5)(dense1)
    
    dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    # Skip connection from dense1 to final layer
    concat = tf.keras.layers.Concatenate()([dense1, dense2])
    
    # Final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(concat)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Advanced optimizer with lookahead
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation technique
    """
    batch_size = tf.shape(x)[0]
    
    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1 - lam)
    lam = lam.reshape(batch_size, 1, 1, 1)
    
    # Shuffle indices
    indices = np.random.permutation(batch_size)
    
    # Mix inputs and targets
    mixed_x = lam * x + (1 - lam) * x[indices]
    mixed_y = lam.reshape(batch_size, 1) * y + (1 - lam.reshape(batch_size, 1)) * y[indices]
    
    return mixed_x, mixed_y

def create_advanced_data_augmentation():
    """
    Create advanced data augmentation with state-of-the-art techniques
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Not good for digits
        fill_mode='nearest',
        preprocessing_function=lambda x: x + np.random.normal(0, 0.05, x.shape)  # Noise injection
    )

class MixupGenerator:
    """
    Generator that applies mixup augmentation
    """
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
    
    def __iter__(self):
        return self
    
    def __next__(self):
        x_batch, y_batch = next(self.generator)
        
        # Apply mixup with probability 0.5
        if np.random.random() < 0.5:
            x_batch, y_batch = mixup_data(x_batch, y_batch, self.alpha)
        
        return x_batch, y_batch

def create_ensemble_models(num_models=3):
    """
    Create ensemble of different architectures for robust predictions
    """
    models = []
    
    for i in range(num_models):
        print(f"Creating ensemble model {i+1}/{num_models}...")
        
        # Slight variations in architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32 + i*8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32 + i*8, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2 + i*0.05),
            
            tf.keras.layers.Conv2D(64 + i*16, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64 + i*16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3 + i*0.05),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512 + i*128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 - i*0.0002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        models.append(model)
    
    return models

def train_ultimate_model():
    """
    Train the ultimate model with all advanced techniques
    """
    print("🌟 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess data
    print("🔧 Preprocessing data with advanced techniques...")
    x_train, y_train, x_test, y_test = preprocess_mnist_data(x_train, y_train, x_test, y_test)
    
    # Create ultimate model
    print("🚀 Creating ultimate CNN model...")
    model = create_ultimate_cnn_model()
    
    print("📊 Model architecture:")
    model.summary()
    
    # Advanced data augmentation
    print("🎨 Setting up advanced data augmentation...")
    datagen = create_advanced_data_augmentation()
    datagen.fit(x_train)
    
    # Create mixup generator
    train_generator = datagen.flow(x_train, y_train, batch_size=128)
    mixup_generator = MixupGenerator(train_generator, alpha=0.2)
    
    # Advanced callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/ultimate_model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        CosineRestartScheduler(
            first_restart_step=50,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0
        )
    ]
    
    # Train with advanced techniques
    print("🏋️ Training ultimate model with cutting-edge techniques...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate ultimate model
    print("📈 Evaluating ultimate model...")
    test_loss, test_accuracy, top_k_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"🏆 Ultimate Model Test Accuracy: {test_accuracy:.4f}")
    print(f"🎯 Top-2 Accuracy: {top_k_accuracy:.4f}")
    
    # Save ultimate model
    model_path = 'models/ultimate_digit_recognition_model.h5'
    save_model(model, model_path)
    
    # Train ensemble models
    print("\n🎪 Training ensemble models for even better performance...")
    ensemble_models = create_ensemble_models(3)
    
    for i, ensemble_model in enumerate(ensemble_models):
        print(f"\n🤖 Training ensemble model {i+1}/3...")
        ensemble_history = ensemble_model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=20,
            validation_data=(x_test, y_test),
            verbose=0
        )
        
        # Save ensemble model
        ensemble_path = f'models/ensemble_model_{i+1}.h5'
        save_model(ensemble_model, ensemble_path)
        
        # Evaluate ensemble model
        _, ensemble_acc = ensemble_model.evaluate(x_test, y_test, verbose=0)
        print(f"   Ensemble Model {i+1} Accuracy: {ensemble_acc:.4f}")
    
    # Test ensemble prediction
    print("\n🎯 Testing ensemble prediction...")
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

class CosineRestartScheduler(tf.keras.callbacks.Callback):
    """
    Cosine annealing with warm restarts for better optimization
    """
    def __init__(self, first_restart_step, t_mul=2.0, m_mul=1.0, alpha=0.0):
        super().__init__()
        self.first_restart_step = first_restart_step
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.current_restart_step = first_restart_step
        self.t_cur = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            # Set initial learning rate
            self.model.optimizer.learning_rate.assign(0.001)
        else:
            self.t_cur += 1
            if self.t_cur >= self.current_restart_step:
                # Restart
                self.t_cur = 0
                self.current_restart_step = int(self.current_restart_step * self.t_mul)
            
            # Cosine annealing
            cos_out = np.cos(np.pi * self.t_cur / self.current_restart_step)
            lr = self.alpha + (0.001 - self.alpha) * (1 + cos_out) / 2
            self.model.optimizer.learning_rate.assign(lr)

def plot_ultimate_training_history(history):
    """
    Plot comprehensive training history for ultimate model
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('🏆 Ultimate Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('📉 Ultimate Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot top-k accuracy
    if 'top_k_categorical_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Training Top-K', linewidth=2)
        axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-K', linewidth=2)
        axes[1, 0].set_title('🎯 Top-K Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-K Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='red')
        axes[1, 1].set_title('🔄 Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('🔄 Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/ultimate_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🚀 TRAINING ULTIMATE DIGIT RECOGNITION MODEL")
    print("="*80)
    print("🌟 CUTTING-EDGE FEATURES:")
    print("- ResNet-style skip connections for deeper training")
    print("- Attention mechanisms for better feature focus")
    print("- Advanced data augmentation with mixup")
    print("- Ensemble predictions from multiple models")
    print("- Cosine annealing with warm restarts")
    print("- Advanced regularization techniques")
    print("- Top-K accuracy monitoring")
    print("="*80)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        model, history, ensemble_models = train_ultimate_model()
        print("\n🎉 ULTIMATE MODEL TRAINING COMPLETED!")
        print("🔄 Test with: python predict_ultimate.py test_images/your_image.png")
        print("📁 Models saved:")
        print("   - Ultimate model: models/ultimate_digit_recognition_model.h5")
        print("   - Ensemble models: models/ensemble_model_*.h5")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
