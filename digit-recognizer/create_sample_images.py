"""
Create sample digit images for testing
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_digit_images():
    """
    Create some sample digit images for testing
    """
    # Create a simple digit "5"
    img_5 = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw digit 5
    cv2.rectangle(img_5, (20, 20), (70, 35), 255, -1)  # Top horizontal line
    cv2.rectangle(img_5, (20, 20), (35, 50), 255, -1)  # Left vertical line
    cv2.rectangle(img_5, (20, 45), (70, 60), 255, -1)  # Middle horizontal line
    cv2.rectangle(img_5, (55, 45), (70, 80), 255, -1)  # Right vertical line
    cv2.rectangle(img_5, (20, 65), (70, 80), 255, -1)  # Bottom horizontal line
    
    # Create a simple digit "3"
    img_3 = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw digit 3
    cv2.rectangle(img_3, (20, 20), (70, 35), 255, -1)  # Top horizontal line
    cv2.rectangle(img_3, (20, 45), (65, 60), 255, -1)  # Middle horizontal line
    cv2.rectangle(img_3, (20, 65), (70, 80), 255, -1)  # Bottom horizontal line
    cv2.rectangle(img_3, (55, 20), (70, 80), 255, -1)  # Right vertical line
    
    # Create a simple digit "7"
    img_7 = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw digit 7
    cv2.rectangle(img_7, (20, 20), (70, 35), 255, -1)  # Top horizontal line
    cv2.line(img_7, (70, 35), (35, 80), 255, 8)        # Diagonal line
    
    # Save the images
    cv2.imwrite('test_images/sample_digit_5.png', img_5)
    cv2.imwrite('test_images/sample_digit_3.png', img_3)
    cv2.imwrite('test_images/sample_digit_7.png', img_7)
    
    print("Sample digit images created:")
    print("- test_images/sample_digit_5.png")
    print("- test_images/sample_digit_3.png")
    print("- test_images/sample_digit_7.png")
    
    # Display the created images
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_5, cmap='gray')
    plt.title('Sample Digit 5')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_3, cmap='gray')
    plt.title('Sample Digit 3')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_7, cmap='gray')
    plt.title('Sample Digit 7')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/sample_created_digits.png')
    plt.show()

if __name__ == "__main__":
    create_sample_digit_images()
