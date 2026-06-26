import cv2
import numpy as np

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Corrects the rotation/skew of an image using image moments or contour detection.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to find the text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
        
    # Get the bounding box of those pixels
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle depending on how minAreaRect behaves
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # If angle is minimal, ignore
    if abs(angle) < 1.0:
        return image
        
    # Get dimensions and center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
