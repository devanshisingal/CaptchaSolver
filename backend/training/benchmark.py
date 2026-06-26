import os
import json
import time
import torch
import cv2
import numpy as np
import pytesseract
import string
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.crnn import CRNN
from preprocessing.pipeline import preprocess_image

# Optional: configure tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

chars = string.ascii_uppercase + string.digits
chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
idx_to_char[0] = '-' 

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(pred, target):
    if len(target) == 0:
        return 1.0
    return levenshtein_distance(pred, target) / len(target)

def decode_prediction(preds):
    preds = preds.argmax(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    char_list = []
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            char_list.append(idx_to_char[preds[i].item()])
    return ''.join(char_list)

def benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_json = os.path.join(base_dir, 'dataset', 'test', 'labels.json')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    model_path = os.path.join(base_dir, 'weights', 'crnn.pt')
    
    if not os.path.exists(test_json):
        print("Test dataset not found.")
        return
        
    with open(test_json, 'r') as f:
        data = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CRNN
    nc, nclass, nh = 1, len(chars) + 1, 256
    model = CRNN(imgH=32, nc=nc, nclass=nclass, nh=nh).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = {
        'crnn': {'correct': 0, 'total': len(data), 'total_cer': 0, 'time': 0},
        'tesseract': {'correct': 0, 'total': len(data), 'total_cer': 0, 'time': 0}
    }

    print(f"Benchmarking {len(data)} images...")
    
    for idx, item in enumerate(data):
        img_path = os.path.join(test_dir, item['image'])
        true_label = item['label']
        
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            continue
            
        # --- CRNN ---
        start_time = time.time()
        processed = preprocess_image(cv_img)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor)
            crnn_pred = decode_prediction(preds)
        crnn_time = time.time() - start_time
        
        results['crnn']['time'] += crnn_time
        if crnn_pred == true_label:
            results['crnn']['correct'] += 1
        results['crnn']['total_cer'] += calculate_cer(crnn_pred, true_label)
        
        # --- Tesseract ---
        start_time = time.time()
        # Tesseract works better on just grayscale/thresholded
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        tess_pred = pytesseract.image_to_string(gray, config='--psm 8 -c tessedit_char_whitelist=' + chars).strip()
        tess_time = time.time() - start_time
        
        results['tesseract']['time'] += tess_time
        if tess_pred == true_label:
            results['tesseract']['correct'] += 1
        results['tesseract']['total_cer'] += calculate_cer(tess_pred, true_label)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(data)}")

    # Print Report
    print("\n--- Benchmark Results ---")
    for name, stats in results.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        avg_cer = stats['total_cer'] / stats['total']
        avg_time_ms = (stats['time'] / stats['total']) * 1000
        throughput = stats['total'] / stats['time']
        
        print(f"\nModel: {name.upper()}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Latency: {avg_time_ms:.2f} ms/prediction")
        print(f"Throughput: {throughput:.2f} req/sec")

if __name__ == "__main__":
    benchmark()
