import base64
import io
import os
import torch
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import string

from models.crnn import CRNN
from preprocessing.pipeline import preprocess_image

app = FastAPI(title="CAPTCHA Vision Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration matching the model
chars = string.ascii_uppercase + string.digits
chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
idx_to_char[0] = '-' # CTC Blank

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nc = 1
nclass = len(chars) + 1
nh = 256
model = CRNN(imgH=32, nc=nc, nclass=nclass, nh=nh).to(device)

model_path = os.path.join(os.path.dirname(__file__), 'weights', 'crnn.pt')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
else:
    print("Warning: Model weights not found. Predictions will be random until trained.")

class PredictRequest(BaseModel):
    image: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

def decode_prediction(preds):
    preds = preds.argmax(2) # [T, batch_size]
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    char_list = []
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            char_list.append(idx_to_char[preds[i].item()])
            
    return ''.join(char_list)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Decode base64
        base64_data = request.image.split(',')[-1] if ',' in request.image else request.image
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
            
        # Preprocess
        processed = preprocess_image(img)
        
        # Convert to tensor and add batch dim
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            preds = model(tensor)
            
            # Simple confidence estimation based on max probabilities
            probs = torch.nn.functional.softmax(preds, dim=2)
            max_probs, _ = probs.max(2)
            confidence = max_probs.mean().item()
            
            prediction = decode_prediction(preds)
            
        return PredictResponse(prediction=prediction, confidence=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
