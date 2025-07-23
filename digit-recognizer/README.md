# Handwritten Digit Recognition Project

This project implements a handwritten digit recognition system using TensorFlow and OpenCV.

## Features
- Train a CNN model on the MNIST dataset
- Real-time digit recognition from camera input
- Image preprocessing with OpenCV
- Model evaluation and visualization

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python train_model.py
```

3. Test the model:
```bash
python test_model.py
```

4. Use real-time recognition:
```bash
python real_time_recognition.py
```

## Project Structure
- `train_model.py` - Train the CNN model on MNIST dataset
- `test_model.py` - Test the trained model
- `real_time_recognition.py` - Real-time digit recognition using camera
- `model_utils.py` - Utility functions for model operations
- `data_preprocessing.py` - Data preprocessing utilities
- `models/` - Directory to save trained models
- `test_images/` - Directory for test images

## Model Architecture
The project uses a Convolutional Neural Network (CNN) with the following layers:
- Conv2D layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Dense layers for classification
