"""
Verify that all required libraries are installed and working
"""
import sys

def check_imports():
    """Check if all required libraries can be imported"""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not found")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not found")
        return False
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn: {sns.__version__}")
    except ImportError:
        print("✗ Seaborn not found")
        return False
    
    return True

def check_gpu():
    """Check if GPU is available for TensorFlow"""
    import tensorflow as tf
    
    print("\nGPU Availability:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("⚠ No GPU found - will use CPU (slower training)")

def main():
    """Main verification function"""
    print("Checking required libraries...")
    print("=" * 40)
    
    if check_imports():
        print("\n✓ All libraries are installed successfully!")
        check_gpu()
        print("\n🚀 Ready to start training!")
        print("\nNext steps:")
        print("1. Run 'python train_model.py' to train the model")
        print("2. Run 'python test_model.py' to test the trained model")
        print("3. Run 'python real_time_recognition.py' for real-time recognition")
    else:
        print("\n✗ Some libraries are missing. Please install them first.")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
