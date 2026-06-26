# CAPTCHA Vision Assistant

A powerful, AI-driven CAPTCHA detection and solving toolkit. This project features a robust **React-based Chrome Extension** that uses DOM heuristics to identify CAPTCHA challenges, and a **FastAPI backend** running a custom PyTorch **CRNN (Convolutional Recurrent Neural Network)** to predict the text.

## 🚀 Features

- **End-to-End Pipeline**: Includes tools for synthetic dataset generation, model training, and real-time inference.
- **Advanced Preprocessing**: Uses OpenCV to deskew, denoise (Median Blur), and binarize (Adaptive Thresholding) distorted images for optimal OCR accuracy.
- **Custom CRNN Model**: A lightweight PyTorch model (CNN + BiLSTM + CTC Decoder) trained specifically on sequence prediction for CAPTCHAs, outperforming traditional OCR tools like Tesseract.
- **Premium Browser Extension**: A Manifest V3 Chrome extension built with React, Tailwind CSS v4, and Framer Motion. Features a dark-mode glassmorphism UI, confidence tracking, and DOM heuristics scoring to accurately locate CAPTCHA elements on any page.
- **Dockerized Deployment**: A production-ready `docker-compose` setup for the FastAPI backend and test environment.

---

## 🏗️ Architecture

```
captcha-vision-assistant/
├── backend/
│   ├── dataset/          # Synthetic data generator (generator.py)
│   ├── models/           # PyTorch CRNN architecture (crnn.py)
│   ├── preprocessing/    # OpenCV pipeline (denoise, deskew, threshold)
│   ├── training/         # Training loop (train.py) & Tesseract benchmarks
│   ├── weights/          # Saved model weights (.pt)
│   ├── api.py            # FastAPI inference server
│   ├── requirements.txt
│   └── Dockerfile.backend
├── extension/            # React + Vite Chrome Extension
│   ├── src/
│   │   ├── content/      # DOM Scanner & Scorer heuristics
│   │   ├── background.ts # Service worker & API communication
│   │   └── App.tsx       # Popup UI
│   └── manifest.json
├── frontend-test/        # Local HTML sandbox for testing the extension
└── docker-compose.yml
```

---

## 🛠️ Getting Started

### 1. Run the Backend & Sandbox Environment

Make sure you have Docker installed.

```bash
# Start the FastAPI backend (port 8000) and the Sandbox Test site (port 8080)
docker-compose up -d --build
```

### 2. Install the Browser Extension

1. Open Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer Mode** on (top right corner).
3. Click **Load unpacked** and select the `/extension/dist` directory in this project.

### 3. Test the System

1. Navigate to **http://localhost:8080** in Chrome. You will see a secure login page with a test CAPTCHA.
2. Click the **CAPTCHA Vision** extension icon in your toolbar.
3. Click **Scan Page**. The extension will highlight the CAPTCHA and display the predicted text, confidence score, and inference latency!

---

## 🧠 Training Your Own Model

If you want to train the model from scratch (instead of using pre-compiled weights):

1. **Setup Python Environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```

2. **Generate the Dataset:**
   ```bash
   # Generates 20,000 training and 2,000 testing synthetic CAPTCHAs
   python dataset/generator.py
   ```

3. **Train the CRNN:**
   ```bash
   # Trains the model using CTC loss and saves the best weights to backend/weights/
   python training/train.py
   ```

4. **Run Benchmarks (Optional):**
   ```bash
   # Compares the custom CRNN against Tesseract OCR (requires tesseract installed on your host)
   python training/benchmark.py
   ```
