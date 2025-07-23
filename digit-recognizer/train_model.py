"""
Train a CNN model for handwritten digit recognition using MNIST dataset
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_mnist_data
from model_utils import create_cnn_model, save_model

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def visualize_sample_data(x_train, y_train, num_samples=25):
    """
    Visualize sample training data
    
    Args:
        x_train: Training images
        y_train: Training labels
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {np.argmax(y_train[i])}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/sample_data.png')
    plt.show()

def main():
    """
    Main training function
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Preprocess the data
    print("Preprocessing data...")
    x_train, y_train, x_test, y_test = preprocess_mnist_data(x_train, y_train, x_test, y_test)
    
    # Visualize sample data
    print("Visualizing sample data...")
    visualize_sample_data(x_train, y_train)
    
    # Create the model
    print("Creating CNN model...")
    model = create_cnn_model()
    
    # Print model summary
    print("Model architecture:")
    model.summary()
    
    # Train the model
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the model
    model_path = 'models/digit_recognition_model.h5'
    save_model(model, model_path)
    
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
