import cv2
import numpy as np

def denoise(image: np.ndarray) -> np.ndarray:
    """
    Removes background noise and artifacts like salt & pepper noise.
    """
    # Median blur is highly effective against salt and pepper noise
    denoised = cv2.medianBlur(image, 3)
    
    return denoised
