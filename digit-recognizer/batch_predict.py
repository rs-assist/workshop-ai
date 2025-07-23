"""
Batch prediction script to test multiple digit images at once
Usage: python batch_predict.py
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_image, extract_digit_from_image
from model_utils import load_model, predict_digit

def batch_predict_images(image_folder='test_images', model_path='models/digit_recognition_model.h5'):
    """
    Predict digits from multiple images in a folder
    
    Args:
        image_folder: Folder containing test images
        model_path: Path to the trained model
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    if model is None:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Get all image files from the folder
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No image files found in {image_folder}/")
        print(f"Supported formats: {supported_formats}")
        return
    
    print(f"Found {len(image_files)} image(s) to process...")
    print("="*60)
    
    results = []
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        print("-" * 40)
        
        # Load and process the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Error: Could not load {image_file}")
            continue
        
        try:
            # Extract digit region
            digit_region = extract_digit_from_image(image)
            
            # Preprocess for model
            preprocessed = preprocess_image(digit_region)
            
            # Make prediction
            predicted_digit, confidence = predict_digit(model, preprocessed)
            
            # Store results
            results.append({
                'file': image_file,
                'prediction': predicted_digit,
                'confidence': confidence
            })
            
            # Display results
            print(f"✅ Predicted: {predicted_digit}")
            print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            if confidence > 0.9:
                print("🟢 High confidence")
            elif confidence > 0.7:
                print("🟡 Medium confidence")
            else:
                print("🔴 Low confidence")
                
        except Exception as e:
            print(f"❌ Error processing {image_file}: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    
    if results:
        # Sort by confidence (highest first)
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        print(f"{'File':<20} {'Prediction':<10} {'Confidence':<12} {'Quality'}")
        print("-" * 60)
        
        for result in results_sorted:
            file_name = result['file'][:18] + ".." if len(result['file']) > 20 else result['file']
            confidence_pct = f"{result['confidence']*100:.1f}%"
            
            if result['confidence'] > 0.9:
                quality = "🟢 High"
            elif result['confidence'] > 0.7:
                quality = "🟡 Medium"
            else:
                quality = "🔴 Low"
            
            print(f"{file_name:<20} {result['prediction']:<10} {confidence_pct:<12} {quality}")
        
        # Statistics
        avg_confidence = np.mean([r['confidence'] for r in results])
        high_conf_count = len([r for r in results if r['confidence'] > 0.9])
        
        print("\n" + "-" * 60)
        print(f"Total images processed: {len(results)}")
        print(f"Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        print(f"High confidence predictions (>90%): {high_conf_count}/{len(results)}")
        
        # Save results to file
        save_results_to_file(results)
        
    else:
        print("No images were successfully processed.")

def save_results_to_file(results):
    """Save results to a text file"""
    with open('models/batch_prediction_results.txt', 'w') as f:
        f.write("Batch Digit Recognition Results\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)\n")
            f.write("-" * 30 + "\n")
    
    print(f"💾 Results saved to: models/batch_prediction_results.txt")

def visualize_batch_results(image_folder='test_images', max_images=9):
    """
    Create a visualization grid of multiple predictions
    """
    # Get image files
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print("No images found for visualization")
        return
    
    # Load model
    model = load_model('models/digit_recognition_model.h5')
    if model is None:
        return
    
    # Limit number of images to display
    image_files = image_files[:max_images]
    
    # Create subplot grid
    rows = int(np.ceil(len(image_files) / 3))
    cols = min(3, len(image_files))
    
    plt.figure(figsize=(15, 5*rows))
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Process image
        digit_region = extract_digit_from_image(image)
        preprocessed = preprocess_image(digit_region)
        predicted_digit, confidence = predict_digit(model, preprocessed)
        
        # Display
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Color based on confidence
        color = 'green' if confidence > 0.9 else 'orange' if confidence > 0.7 else 'red'
        plt.title(f'{image_file}\nPrediction: {predicted_digit}\nConf: {confidence:.3f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/batch_predictions_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Visualization saved to: models/batch_predictions_grid.png")

def main():
    """
    Main function
    """
    print("🔍 BATCH DIGIT RECOGNITION")
    print("="*40)
    
    # Run batch prediction
    batch_predict_images()
    
    # Ask if user wants visualization
    print("\n📊 Would you like to see a visual grid of all predictions?")
    print("This will create a plot showing all images with their predictions.")
    
    # Create visualization
    visualize_batch_results()

if __name__ == "__main__":
    main()
