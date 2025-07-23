"""
Create sample hand-drawn digit images for testing
"""
import cv2
import numpy as np

def create_sample_digits():
    """Create sample digit images for testing"""
    
    # Create a white canvas
    def create_canvas():
        return np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Sample digit 3
    img3 = create_canvas()
    cv2.putText(img3, '3', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
    cv2.imwrite('test_images/sample_digit_3.png', img3)
    
    # Sample digit 7
    img7 = create_canvas()
    cv2.putText(img7, '7', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
    cv2.imwrite('test_images/sample_digit_7.png', img7)
    
    # Sample digit 5
    img5 = create_canvas()
    cv2.putText(img5, '5', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
    cv2.imwrite('test_images/sample_digit_5.png', img5)
    
    print("Sample digit images created:")
    print("- test_images/sample_digit_3.png")
    print("- test_images/sample_digit_7.png")
    print("- test_images/sample_digit_5.png")

if __name__ == "__main__":
    create_sample_digits()
