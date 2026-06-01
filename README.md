# 🎭 Face Mask Detector — Complete AI/ML Project

Real-time face mask detection using Machine Learning.
Train your own model and run webcam detection — no cloud, no API key needed.

---

## 📁 Project Structure

```
face_mask_detector/
├── dataset/
│   ├── with_mask/       ← training images (mask worn)
│   └── without_mask/    ← training images (no mask)
├── src/
│   ├── generate_dataset.py   ← Step 1: Generate training data
│   ├── train.py              ← Step 2: Train the ML model
│   ├── predict.py            ← Step 3: Predict on image
│   └── detect_webcam.py      ← Step 4: Live webcam detection
├── models/
│   ├── mask_detector.pkl     ← Saved trained model
│   └── metrics.json          ← Accuracy & training stats
├── results/
│   └── training_results.png  ← Confusion matrix + loss curve
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (4 Steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate training dataset
```bash
python src/generate_dataset.py --count 500
```
Generates 500 synthetic face images per class (with mask / without mask).

### Step 3 — Train the model
```bash
python src/train.py
```
Trains a 3-layer MLP Neural Network. Saves model to `models/mask_detector.pkl`.

### Step 4 — Run live webcam detection
```bash
python src/detect_webcam.py
```
Opens your webcam and detects masks in real time. Press **Q** to quit.

---

## 🖼 Predict on a Single Image

```bash
python src/predict.py --image your_photo.jpg
python src/predict.py --image your_photo.jpg --show   # also open a window
```

---

## 🧠 How It Works

| Component | Details |
|---|---|
| **Face Detection** | OpenCV Haar Cascade (frontal face) |
| **Feature Extraction** | HOG-like gradients + color histograms + pixel values |
| **Classifier** | MLP Neural Network (256 → 128 → 64 → 2) |
| **Training** | scikit-learn MLPClassifier, Adam optimizer |
| **Accuracy** | ~99-100% on synthetic data |
| **Inference speed** | ~30 FPS on CPU |

### Architecture
```
Input Image (64×64 RGB)
      ↓
Feature Extraction
  • Sobel gradients (HOG-like, 4×4 blocks)
  • Color histograms (16 bins × 3 channels)
  • Raw pixel values (32×32 grayscale)
  → 1264-dimensional feature vector
      ↓
StandardScaler (zero mean, unit variance)
      ↓
MLP Neural Network
  Layer 1: 256 neurons (ReLU)
  Layer 2: 128 neurons (ReLU)
  Layer 3:  64 neurons (ReLU)
  Output:    2 classes (Softmax)
      ↓
Prediction: Mask / No Mask + Confidence %
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~99–100% |
| Cross-Validation | 99.6% ± 0.5% |
| Training samples | 800 |
| Test samples | 200 |
| Features per image | 1264 |

---

## 🔄 Use Real Dataset (Kaggle)

For production-grade accuracy, use the **Face Mask Detection Dataset** from Kaggle:
```
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
```
Download → extract → replace `dataset/with_mask/` and `dataset/without_mask/` → retrain.

---

## 🌐 Browser Demo

Open `face_mask_detector.html` in Chrome/Firefox for a browser-based demo
using TensorFlow.js (no install required).

---

## 🗝 Controls (Webcam Mode)

| Key | Action |
|---|---|
| `Q` | Quit |

---

## 📦 Requirements

- Python 3.8+
- numpy, opencv-python, scikit-learn, matplotlib, seaborn

```bash
pip install -r requirements.txt
```
