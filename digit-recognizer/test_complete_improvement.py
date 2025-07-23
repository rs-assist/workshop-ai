#!/usr/bin/env python3
"""
Complete testing of improved model with enhanced preprocessing
This script tests both original and improved models on all test images
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from improved_preprocessing import improved_preprocess_image

def load_models():
    """Load both original and improved models"""
    print("Loading models...")
    
    # Load original model
    original_model = keras.models.load_model('models/digit_recognition_model.h5')
    
    # Load improved model
    improved_model = keras.models.load_model('models/improved_digit_recognition_model.h5')
    
    return original_model, improved_model

def original_preprocess_image(image_path):
    """Original preprocessing method for comparison"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize to 0-1
    img = img.astype('float32') / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img

def predict_with_both_methods(image_path, original_model, improved_model):
    """Test image with both original and improved preprocessing + models"""
    
    # Original method (original preprocessing + original model)
    try:
        original_img = original_preprocess_image(image_path)
        original_prediction = original_model.predict(original_img, verbose=0)
        original_digit = np.argmax(original_prediction)
        original_confidence = float(np.max(original_prediction)) * 100
    except Exception as e:
        original_digit = "ERROR"
        original_confidence = 0
    
    # Improved method (enhanced preprocessing + improved model)
    try:
        improved_img = improved_preprocess_image(cv2.imread(image_path))
        improved_prediction = improved_model.predict(improved_img, verbose=0)
        improved_digit = np.argmax(improved_prediction)
        improved_confidence = float(np.max(improved_prediction)) * 100
    except Exception as e:
        improved_digit = "ERROR"
        improved_confidence = 0
    
    return {
        'original': {'digit': original_digit, 'confidence': original_confidence},
        'improved': {'digit': improved_digit, 'confidence': improved_confidence}
    }

def test_all_images():
    """Test all images in the test_images folder"""
    
    # Load models
    original_model, improved_model = load_models()
    
    # Get all test images
    test_folder = 'test_images'
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found!")
        return
    
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No image files found in '{test_folder}'!")
        return
    
    print(f"\n🧪 TESTING {len(image_files)} IMAGES")
    print("="*80)
    
    results = []
    high_confidence_original = 0
    high_confidence_improved = 0
    
    for i, filename in enumerate(sorted(image_files), 1):
        image_path = os.path.join(test_folder, filename)
        
        print(f"\n📷 Image {i}: {filename}")
        print("-" * 40)
        
        # Test with both methods
        result = predict_with_both_methods(image_path, original_model, improved_model)
        
        # Display results
        orig = result['original']
        impr = result['improved']
        
        print(f"🔶 Original Method:")
        print(f"   Predicted: {orig['digit']} (Confidence: {orig['confidence']:.2f}%)")
        
        print(f"🔷 Improved Method:")
        print(f"   Predicted: {impr['digit']} (Confidence: {impr['confidence']:.2f}%)")
        
        # Calculate improvement
        if orig['confidence'] > 0 and impr['confidence'] > 0:
            improvement = impr['confidence'] - orig['confidence']
            print(f"📈 Improvement: {improvement:+.2f}%")
            
            if improvement > 0:
                print("✅ BETTER with improved method!")
            elif improvement < 0:
                print("⚠️  Original method was better")
            else:
                print("🔄 Same performance")
        
        # Count high confidence predictions
        if orig['confidence'] >= 80:
            high_confidence_original += 1
        if impr['confidence'] >= 80:
            high_confidence_improved += 1
        
        results.append({
            'filename': filename,
            'original': orig,
            'improved': impr
        })
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    total_images = len(results)
    print(f"Total images tested: {total_images}")
    print(f"High confidence (≥80%) - Original: {high_confidence_original}/{total_images}")
    print(f"High confidence (≥80%) - Improved: {high_confidence_improved}/{total_images}")
    
    # Calculate average improvements
    valid_results = [r for r in results if r['original']['confidence'] > 0 and r['improved']['confidence'] > 0]
    if valid_results:
        avg_original = np.mean([r['original']['confidence'] for r in valid_results])
        avg_improved = np.mean([r['improved']['confidence'] for r in valid_results])
        overall_improvement = avg_improved - avg_original
        
        print(f"\nAverage confidence - Original: {avg_original:.2f}%")
        print(f"Average confidence - Improved: {avg_improved:.2f}%")
        print(f"Overall improvement: {overall_improvement:+.2f}%")
        
        if overall_improvement > 0:
            print("🎉 IMPROVED METHOD IS SIGNIFICANTLY BETTER!")
        else:
            print("🤔 Results are mixed")
    
    # Show biggest improvements
    improvements = []
    for r in valid_results:
        improvement = r['improved']['confidence'] - r['original']['confidence']
        improvements.append((r['filename'], improvement))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 3 BIGGEST IMPROVEMENTS:")
    for i, (filename, improvement) in enumerate(improvements[:3], 1):
        print(f"{i}. {filename}: {improvement:+.2f}%")
    
    return results

if __name__ == "__main__":
    print("🚀 COMPLETE MODEL IMPROVEMENT TEST")
    print("Testing both original and improved models with enhanced preprocessing")
    print("="*80)
    
    try:
        results = test_all_images()
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
