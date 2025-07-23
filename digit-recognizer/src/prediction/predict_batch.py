#!/usr/bin/env python3
"""
Batch image prediction module
"""

import os
import glob
from .predict_single import load_all_models, predict_digit

def predict_batch_images(folder_path):
    """Predict all images in a folder"""
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print("BATCH DIGIT RECOGNITION")
    print("=" * 50)
    print(f"Processing {len(image_files)} images from {folder_path}")
    
    # Load models once
    models, model_names = load_all_models()
    if not models:
        print("No models found!")
        return
    
    print(f"Using {len(models)} models for ensemble prediction")
    print("=" * 50)
    
    # Process all images
    results = []
    high_confidence_count = 0
    
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing {os.path.basename(image_path)}")
        
        result = predict_digit(models, model_names, image_path)
        if result:
            pred, conf = result
            results.append({
                'file': os.path.basename(image_path),
                'prediction': pred,
                'confidence': conf
            })
            
            if conf >= 0.75:
                high_confidence_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    
    if results:
        total_images = len(results)
        avg_confidence = sum(r['confidence'] for r in results) / total_images
        
        print(f"Total Images Processed: {total_images}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"High Confidence (>=75%): {high_confidence_count}/{total_images} ({high_confidence_count/total_images*100:.1f}%)")
        
        print("\nDETAILED RESULTS:")
        for result in results:
            conf_level = "HIGH" if result['confidence'] >= 0.75 else "MED" if result['confidence'] >= 0.60 else "LOW"
            print(f"   [{conf_level}] {result['file']:20} -> {result['prediction']} ({result['confidence']:.1%})")
        
        # Save results to file
        output_file = os.path.join("outputs", "batch_results.txt")
        os.makedirs("outputs", exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("BATCH PREDICTION RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Average Confidence: {avg_confidence:.1%}\n")
            f.write(f"High Confidence Rate: {high_confidence_count/total_images*100:.1f}%\n\n")
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for result in results:
                f.write(f"{result['file']:20} -> {result['prediction']} ({result['confidence']:.1%})\n")
        
        print(f"\nResults saved to: {output_file}")
    
    return results
