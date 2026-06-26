import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import string
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.crnn import CRNN
from preprocessing.pipeline import preprocess_image

chars = string.ascii_uppercase + string.digits
chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
idx_to_char[0] = '-' # CTC Blank

class CaptchaDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        
        # Load image with Pillow, but we want to apply our preprocessing pipeline first
        import cv2
        import numpy as np
        
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            # Fallback if image not found
            cv_img = np.ones((60, 160, 3), dtype=np.uint8) * 255
            
        processed = preprocess_image(cv_img)
        # Convert to torch tensor [C, H, W]
        tensor_img = torch.from_numpy(processed).unsqueeze(0)
        
        label = item['label']
        label_encoded = [char_to_idx[c] for c in label]
        
        return tensor_img, torch.tensor(label_encoded, dtype=torch.long)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad labels
    target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    max_len = target_lengths.max()
    padded_labels = []
    for lbl in labels:
        padded = torch.cat([lbl, torch.zeros(max_len - len(lbl), dtype=torch.long)])
        padded_labels.append(padded)
        
    labels = torch.stack(padded_labels, 0)
    return images, labels, target_lengths

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    train_json = os.path.join(dataset_dir, 'train', 'labels.json')
    train_dir = os.path.join(dataset_dir, 'train')
    
    if not os.path.exists(train_json):
        print("Dataset not found. Please run generator.py first.")
        return

    dataset = CaptchaDataset(train_json, train_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    nc = 1 # 1 channel (grayscale) after preprocessing
    nclass = len(chars) + 1 # +1 for CTC blank
    nh = 256 # hidden size for LSTM
    
    model = CRNN(imgH=32, nc=nc, nclass=nclass, nh=nh).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, labels, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            # preds is [T, batch_size, num_classes]
            T = preds.size(0)
            batch_size = preds.size(1)
            
            input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long)
            
            loss = criterion(preds.log_softmax(2), labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(weights_dir, 'crnn.pt'))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
