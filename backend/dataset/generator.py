import os
import random
import string
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse

def generate_random_text(length=5):
    """Generate random alphanumeric string."""
    chars = string.ascii_uppercase + string.digits
    # avoiding easily confused chars if possible, but let's just use all for standard CAPTCHA
    chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    return ''.join(random.choices(chars, k=length))

def add_noise(image, noise_type="gaussian"):
    """Add noise to the image."""
    img_array = np.array(image)
    if noise_type == "gaussian":
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img_array.shape)
        gauss = gauss.reshape(img_array.shape)
        noisy = img_array + gauss * 255
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(img_array)
        # Salt
        num_salt = np.ceil(amount * img_array.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
        noisy[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
        noisy[tuple(coords)] = 0
        return Image.fromarray(noisy)
    return image

def apply_distortion(image):
    """Apply wave or perspective distortion."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    distortion_type = random.choice(["wave", "perspective", "none"])
    
    if distortion_type == "wave":
        # Wave distortion
        map_x, map_y = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        for i in range(h):
            for j in range(w):
                map_x[i, j] = j + 3 * np.sin(i / 5.0)
                map_y[i, j] = i + 3 * np.cos(j / 5.0)
        distorted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return Image.fromarray(distorted)
        
    elif distortion_type == "perspective":
        # Perspective skew
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = random.randint(0, h // 4)
        pts2 = np.float32([[0, offset], [w, 0], [offset, h], [w, h - offset]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        distorted = cv2.warpPerspective(img_array, matrix, (w, h), borderValue=(255, 255, 255))
        return Image.fromarray(distorted)
        
    return image

def create_captcha(text, width=160, height=64):
    """Generate a single CAPTCHA image."""
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Try to load a reasonable font, fallback to default
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 40)
    except IOError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()

    # Draw text with random character spacing and slight Y offsets
    x = 10
    for char in text:
        y = random.randint(5, 15)
        
        # Draw individual character onto a temporary image to rotate it independently
        char_img = Image.new('RGBA', (40, 50), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
        
        # Random rotation (-20 to 20 degrees)
        angle = random.randint(-20, 20)
        char_img = char_img.rotate(angle, expand=1, fillcolor=(255, 255, 255, 0))
        
        image.paste(char_img, (x, y), char_img)
        x += random.randint(25, 30) # Advance x

    # Add noise
    noise_type = random.choice(["gaussian", "salt_pepper", "none"])
    if noise_type != "none":
        image = add_noise(image, noise_type)

    # Apply distortion
    image = apply_distortion(image)

    # Add random lines for more obfuscation (common in CAPTCHAs)
    draw = ImageDraw.Draw(image)
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line(((x1, y1), (x2, y2)), fill=(50, 50, 50), width=random.randint(1, 2))

    return image

def generate_dataset(num_images, output_dir, prefix="img"):
    os.makedirs(output_dir, exist_ok=True)
    labels = []

    for i in range(num_images):
        text = generate_random_text()
        img = create_captcha(text)
        
        filename = f"{prefix}_{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        labels.append({
            "image": filename,
            "label": text
        })
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_images} in {output_dir}")

    # Save labels
    labels_file = os.path.join(output_dir, "labels.json")
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"Finished generating {num_images} images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CAPTCHA dataset")
    parser.add_argument("--num_train", type=int, default=20000, help="Number of training images")
    parser.add_argument("--num_test", type=int, default=2000, help="Number of testing images")
    parser.add_argument("--output_dir", type=str, default=".", help="Base output directory")
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    print(f"Generating {args.num_train} training images...")
    generate_dataset(args.num_train, train_dir, prefix="train")
    
    print(f"Generating {args.num_test} testing images...")
    generate_dataset(args.num_test, test_dir, prefix="test")
