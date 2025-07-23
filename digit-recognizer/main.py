#!/usr/bin/env python3
"""
🚀 HANDWRITTEN DIGIT RECOGNITION - MAIN ENTRY POINT
================================================================================
Ultimate AI-powered digit recognition system with ensemble models

Usage:
    python main.py predict <image_path>          # Predict single image
    python main.py batch <folder_path>           # Batch prediction
    python main.py realtime                      # Real-time webcam recognition
    python main.py train                         # Train new models

Features:
    ✅ Ensemble prediction with multiple models
    ✅ High-confidence predictions (70-100%)
    ✅ Real-time webcam processing
    ✅ Batch processing capabilities
    ✅ Advanced preprocessing pipeline
================================================================================
"""

import sys
import os
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(
        description='Ultimate Handwritten Digit Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py predict data/test_images/test1.png
    python main.py batch data/test_images/
    python main.py realtime
    python main.py train
    python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict single image')
    predict_parser.add_argument('image_path', help='Path to image file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch prediction')
    batch_parser.add_argument('folder_path', help='Path to folder with images')
    
    # Real-time command
    subparsers.add_parser('realtime', help='Real-time webcam recognition')
    
    # Training command
    subparsers.add_parser('train', help='Train new models')
    
    # Test command
    subparsers.add_parser('test', help='Run test suite')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'predict':
            from prediction.predict_single import predict_single_image
            predict_single_image(args.image_path)
            
        elif args.command == 'batch':
            from prediction.predict_batch import predict_batch_images
            predict_batch_images(args.folder_path)
            
        elif args.command == 'realtime':
            from real_time.webcam_recognition import start_realtime_recognition
            start_realtime_recognition()
            
        elif args.command == 'train':
            from training.train_ultimate import train_ultimate_models
            train_ultimate_models()

    except ImportError as e:
        print(f"ERROR: Error importing module: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
