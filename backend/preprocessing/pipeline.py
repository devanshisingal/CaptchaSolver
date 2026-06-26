import cv2
import numpy as np

from .denoise import denoise
from .deskew import deskew
from .threshold import apply_threshold

def preprocess_image(image: np.ndarray, target_size=(160, 32)) -> np.ndarray:
    """
    Executes the full preprocessing pipeline on an input image.
    Pipeline: grayscale -> denoise -> deskew -> normalize/resize -> threshold
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Denoise
    denoised = denoise(gray)
    
    # 3. Deskew
    deskewed = deskew(denoised)
    
    # 4. Resize to target size for the model
    resized = cv2.resize(deskewed, target_size, interpolation=cv2.INTER_AREA)
    
    # 5. Threshold (binarize)
    # Using adaptive thresholding or Otsu's depending on testing. We'll use our threshold.py
    # actually our apply_threshold function handles grayscale check, so we can just pass resized
    binary = apply_threshold(resized)
    
    # Return as single channel float32 array normalized to 0-1 for PyTorch
    normalized = binary.astype(np.float32) / 255.0
    return normalized
