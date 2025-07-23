#!/usr/bin/env python3
"""
🚀 ULTIMATE SIMPLE PREDICTION SYSTEM
================================================================================
A highly optimized prediction system that combines:
- Multiple trained models for ensemble predictions
- Advanced preprocessing
- Confidence analysis
- Uncertainty estimation
================================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import sys
from improved_preprocessing import preprocess_image

def load_available_models():
    """
    Load all available trained models for ensemble prediction
    """
    models = []
    model_names = []
    
    # Check for ultimate simple model
    if os.path.exists('models/ultimate_simple_model.keras'):
        try:
            model = keras.models.load_model('models/ultimate_simple_model.keras')
            models.append(model)
            model_names.append('Ultimate Simple')
            print("✅ Loaded Ultimate Simple Model")
        except Exception as e:
            print(f"⚠️ Could not load ultimate simple model: {e}")
    
    # Check for ensemble models
    for i in range(1, 4):
        model_path = f'models/ultimate_ensemble_{i}.keras'
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                models.append(model)
                model_names.append(f'Ensemble {i}')
                print(f"✅ Loaded Ensemble Model {i}")
            except Exception as e:
                print(f"⚠️ Could not load ensemble model {i}: {e}")
    
    # Check for improved model
    if os.path.exists('models/improved_digit_model.keras'):
        try:
            model = keras.models.load_model('models/improved_digit_model.keras')
            models.append(model)
            model_names.append('Improved')
            print("✅ Loaded Improved Model")
        except Exception as e:
            print(f"⚠️ Could not load improved model: {e}")
    
    # Check for original model
    if os.path.exists('models/digit_recognition_model.keras'):
        try:
            model = keras.models.load_model('models/digit_recognition_model.keras')
            models.append(model)
            model_names.append('Original')
            print("✅ Loaded Original Model")
        except Exception as e:
            print(f"⚠️ Could not load original model: {e}")
    
    if not models:
        print("❌ No models found! Please train a model first.")
        return None, None
    
    print(f"🏆 Total models loaded: {len(models)}")
    return models, model_names

def test_time_augmentation(image, num_augmentations=5):
    """
    Apply test-time augmentation for better predictions
    """
    augmented_images = [image]  # Original image
    
    for _ in range(num_augmentations - 1):
        # Random small rotations
        angle = np.random.uniform(-5, 5)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        
        # Small translations
        tx = np.random.uniform(-2, 2)
        ty = np.random.uniform(-2, 2)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(rotated, translation_matrix, (cols, rows))
        
        augmented_images.append(translated)
    
    return np.array(augmented_images)

def ensemble_predict_with_confidence(models, model_names, image_path):
    """
    Predict digit using ensemble of models with confidence analysis
    """
    print(f"\n🔍 Analyzing image: {image_path}")
    print("=" * 50)\n    \n    # Preprocess image\n    processed_image = preprocess_image(image_path)\n    if processed_image is None:\n        return None\n    \n    # Apply test-time augmentation\n    augmented_images = test_time_augmentation(processed_image)\n    \n    all_predictions = []\n    individual_results = []\n    \n    for i, (model, name) in enumerate(zip(models, model_names)):\n        model_predictions = []\n        \n        # Predict on all augmented versions\n        for aug_image in augmented_images:\n            pred = model.predict(aug_image.reshape(1, 28, 28, 1), verbose=0)\n            model_predictions.append(pred[0])\n        \n        # Average predictions across augmentations\n        avg_pred = np.mean(model_predictions, axis=0)\n        all_predictions.append(avg_pred)\n        \n        predicted_digit = np.argmax(avg_pred)\n        confidence = avg_pred[predicted_digit]\n        \n        individual_results.append({\n            'model': name,\n            'prediction': predicted_digit,\n            'confidence': confidence,\n            'probabilities': avg_pred\n        })\n        \n        print(f\"📊 {name:15} → Digit: {predicted_digit}, Confidence: {confidence:.1%}\")\n    \n    # Ensemble prediction\n    ensemble_avg = np.mean(all_predictions, axis=0)\n    ensemble_prediction = np.argmax(ensemble_avg)\n    ensemble_confidence = ensemble_avg[ensemble_prediction]\n    \n    # Calculate uncertainty (standard deviation across models)\n    pred_std = np.std([pred[ensemble_prediction] for pred in all_predictions])\n    uncertainty = pred_std / ensemble_confidence if ensemble_confidence > 0 else 1.0\n    \n    # Voting-based prediction\n    votes = [result['prediction'] for result in individual_results]\n    unique_votes, vote_counts = np.unique(votes, return_counts=True)\n    voting_prediction = unique_votes[np.argmax(vote_counts)]\n    voting_agreement = np.max(vote_counts) / len(votes)\n    \n    print(\"\\n\" + \"=\" * 50)\n    print(\"🏆 ENSEMBLE RESULTS:\")\n    print(f\"📊 Average Prediction: {ensemble_prediction} (Confidence: {ensemble_confidence:.1%})\")\n    print(f\"🗳️  Voting Prediction: {voting_prediction} (Agreement: {voting_agreement:.1%})\")\n    print(f\"📈 Uncertainty Score: {uncertainty:.3f} (lower is better)\")\n    \n    # Final prediction (prefer voting if high agreement, otherwise use ensemble average)\n    final_prediction = voting_prediction if voting_agreement >= 0.6 else ensemble_prediction\n    final_confidence = voting_agreement if voting_agreement >= 0.6 else ensemble_confidence\n    \n    print(f\"\\n🎯 FINAL PREDICTION: {final_prediction}\")\n    print(f\"🎯 FINAL CONFIDENCE: {final_confidence:.1%}\")\n    \n    # Confidence level assessment\n    if final_confidence >= 0.95:\n        confidence_level = \"🟢 VERY HIGH\"\n    elif final_confidence >= 0.85:\n        confidence_level = \"🟡 HIGH\"\n    elif final_confidence >= 0.70:\n        confidence_level = \"🟠 MEDIUM\"\n    else:\n        confidence_level = \"🔴 LOW\"\n    \n    print(f\"📊 Confidence Level: {confidence_level}\")\n    \n    # Show top 3 predictions\n    top_3_indices = np.argsort(ensemble_avg)[-3:][::-1]\n    print(\"\\n🏅 Top 3 Predictions:\")\n    for j, idx in enumerate(top_3_indices):\n        print(f\"   {j+1}. Digit {idx}: {ensemble_avg[idx]:.1%}\")\n    \n    return {\n        'prediction': final_prediction,\n        'confidence': final_confidence,\n        'ensemble_avg': ensemble_avg,\n        'individual_results': individual_results,\n        'uncertainty': uncertainty,\n        'voting_agreement': voting_agreement\n    }\n\ndef main():\n    if len(sys.argv) < 2:\n        print(\"Usage: python predict_ultimate_simple.py <image_path>\")\n        print(\"Example: python predict_ultimate_simple.py test_images/test1.png\")\n        return\n    \n    image_path = sys.argv[1]\n    \n    if not os.path.exists(image_path):\n        print(f\"❌ Image file not found: {image_path}\")\n        return\n    \n    print(\"🚀 ULTIMATE SIMPLE PREDICTION SYSTEM\")\n    print(\"=\" * 50)\n    \n    # Load models\n    models, model_names = load_available_models()\n    if models is None:\n        return\n    \n    # Make prediction\n    result = ensemble_predict_with_confidence(models, model_names, image_path)\n    \n    if result:\n        print(\"\\n✅ Prediction completed successfully!\")\n    else:\n        print(\"\\n❌ Prediction failed!\")\n\nif __name__ == \"__main__\":\n    main()\n
