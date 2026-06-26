import cv2
import numpy as np

def apply_threshold(image: np.ndarray) -> np.ndarray:
    """
    Applies adaptive thresholding to convert the image to binary and highlight text.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding is robust against varying lighting/backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    return binary
