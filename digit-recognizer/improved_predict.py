"""
Improved prediction script with better preprocessing
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from improved_preprocessing import improved_preprocess_image, extract_digit_improved
from model_utils import load_model, predict_digit

def predict_with_improved_preprocessing(image_path, use_improved_model=False):
    """
    Predict digit using improved preprocessing techniques
    
    Args:
        image_path: Path to the image file
        use_improved_model: Whether to use the improved model if available
    """
    # Choose model
    if use_improved_model:
        model_path = 'models/improved_digit_recognition_model.h5'
        print(f"🚀 Using improved model: {model_path}")
    else:
        model_path = 'models/digit_recognition_model.h5'
        print(f"📊 Using original model: {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        if use_improved_model:
            print("❌ Improved model not found. Train it first with: python train_improved_model.py")
            print("🔄 Falling back to original model...")
            return predict_with_improved_preprocessing(image_path, use_improved_model=False)
        else:
            print("❌ No model found. Please train a model first.")
            return
    
    # Load image
    print(f"📁 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print("🔧 Processing with improved preprocessing...")
    
    # Method 1: Original preprocessing
    print("\n--- Method 1: Original Preprocessing ---")
    from data_preprocessing import extract_digit_from_image, preprocess_image
    
    digit_region_orig = extract_digit_from_image(image)
    preprocessed_orig = preprocess_image(digit_region_orig)
    predicted_orig, confidence_orig = predict_digit(model, preprocessed_orig)
    
    print(f"Original: Digit {predicted_orig}, Confidence: {confidence_orig:.4f} ({confidence_orig*100:.2f}%)")
    
    # Method 2: Improved preprocessing
    print("\n--- Method 2: Improved Preprocessing ---")
    digit_region_improved = extract_digit_improved(image)
    preprocessed_improved = improved_preprocess_image(digit_region_improved)
    predicted_improved, confidence_improved = predict_digit(model, preprocessed_improved)
    
    print(f"Improved: Digit {predicted_improved}, Confidence: {confidence_improved:.4f} ({confidence_improved*100:.2f}%)")
    
    # Compare results
    print("\n" + "="*60)
    print("🏆 COMPARISON RESULTS")
    print("="*60)
    
    if confidence_improved > confidence_orig:
        improvement = confidence_improved - confidence_orig
        print(f"✅ Improved preprocessing is better!")
        print(f"📈 Confidence improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
        best_method = "Improved"
        best_prediction = predicted_improved
        best_confidence = confidence_improved
        best_preprocessed = preprocessed_improved
    else:
        decline = confidence_orig - confidence_improved
        print(f"⚠️  Original preprocessing performed better")
        print(f"📉 Confidence decline: -{decline:.4f} ({decline*100:.2f}%)")
        best_method = "Original"
        best_prediction = predicted_orig
        best_confidence = confidence_orig
        best_preprocessed = preprocessed_orig
    
    print(f"\n🎯 Best Result: Digit {best_prediction} with {best_confidence:.4f} confidence ({best_method} method)")
    
    # Visualize comparison
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Original preprocessing steps
    plt.subplot(2, 4, 2)
    plt.imshow(digit_region_orig, cmap='gray')
    plt.title('Original: Extracted')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(preprocessed_orig.reshape(28, 28), cmap='gray')
    plt.title(f'Original: Final\nPred: {predicted_orig} ({confidence_orig:.3f})')
    plt.axis('off')
    
    # Improved preprocessing steps  
    plt.subplot(2, 4, 5)
    plt.imshow(digit_region_improved, cmap='gray')
    plt.title('Improved: Extracted')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(preprocessed_improved.reshape(28, 28), cmap='gray')
    plt.title(f'Improved: Final\nPred: {predicted_improved} ({confidence_improved:.3f})')
    plt.axis('off')
    
    # Show detailed preprocessing steps for improved method
    plt.subplot(2, 4, 7)
    improved_preprocess_image(digit_region_improved, show_steps=False)
    
    # Best result highlight
    plt.subplot(2, 4, 8)
    plt.imshow(best_preprocessed.reshape(28, 28), cmap='gray')
    color = 'green' if best_confidence > 0.9 else 'orange' if best_confidence > 0.7 else 'red'
    plt.title(f'🏆 BEST RESULT\nDigit: {best_prediction}\nConf: {best_confidence:.3f}', color=color)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'models/improved_prediction_{image_path.split("/")[-1]}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_prediction, best_confidence, best_method

def main():
    """
    Main function
    """
    if len(sys.argv) < 2:
        print("Usage: python improved_predict.py <image_path> [use_improved_model]")
        print("Examples:")
        print("  python improved_predict.py test_images/test1.png")
        print("  python improved_predict.py test_images/test1.png true")
        return
    
    image_path = sys.argv[1]
    use_improved_model = len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'yes']
    
    print("🔍 IMPROVED DIGIT RECOGNITION")
    print("="*50)
    
    predict_with_improved_preprocessing(image_path, use_improved_model)

if __name__ == "__main__":
    main()
