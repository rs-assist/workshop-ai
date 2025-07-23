"""
Test the trained digit recognition model
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_mnist_data
from model_utils import load_model, predict_digit

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    Visualize model predictions on test data
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels
        num_samples: Number of samples to display
    """
    # Get random indices
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        
        # Get the image and true label
        image = x_test[idx]
        true_label = np.argmax(y_test[idx])
        
        # Make prediction
        predicted_digit, confidence = predict_digit(model, image.reshape(1, 28, 28, 1))
        
        # Display image
        plt.imshow(image.reshape(28, 28), cmap='gray')
        
        # Set title with prediction and confidence
        color = 'green' if predicted_digit == true_label else 'red'
        plt.title(f'True: {true_label}, Pred: {predicted_digit}\nConf: {confidence:.3f}', 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/test_predictions.png')
    plt.show()

def calculate_confusion_matrix(model, x_test, y_test):
    """
    Calculate and display confusion matrix
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Make predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))

def test_model_performance(model, x_test, y_test):
    """
    Test model performance on test dataset
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels
    """
    print("Testing model performance...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")
    
    return test_loss, test_accuracy

def main():
    """
    Main testing function
    """
    print("Loading MNIST test dataset...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    print("Preprocessing data...")
    x_train, y_train, x_test, y_test = preprocess_mnist_data(x_train, y_train, x_test, y_test)
    
    # Load the trained model
    model_path = 'models/digit_recognition_model.h5'
    print(f"Loading model from {model_path}...")
    
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first by running train_model.py")
        return
    
    # Test model performance
    test_loss, test_accuracy = test_model_performance(model, x_test, y_test)
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, x_test, y_test)
    
    # Calculate confusion matrix
    print("Calculating confusion matrix...")
    calculate_confusion_matrix(model, x_test, y_test)
    
    print(f"\nTesting completed!")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
