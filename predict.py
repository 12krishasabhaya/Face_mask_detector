"""
Face Mask Detector — Predict on Image
=======================================
Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/image.jpg --show
"""

import numpy as np
import cv2
import pickle
import os
import argparse


MODEL_PATH = "models/mask_detector.pkl"
IMG_SIZE   = 64
FEAT_SIZE  = 32


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun: python src/train.py")
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['metrics']


def extract_features(img):
    """Same feature extraction as training."""
    img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (FEAT_SIZE, FEAT_SIZE))

    gx  = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    feats = []
    for r in range(0, FEAT_SIZE, 8):
        for c in range(0, FEAT_SIZE, 8):
            bm = mag[r:r+8, c:c+8].flatten()
            ba = ang[r:r+8, c:c+8].flatten()
            feats += [bm.mean(), bm.std(), ba.mean(), ba.std()]
            hist, _ = np.histogram(ba, bins=8, range=(-np.pi, np.pi), weights=bm)
            feats += list(hist)
    for ch in range(3):
        h = cv2.calcHist([img], [ch], None, [16], [0, 256]).flatten()
        feats += list(h)
    feats += list(small.flatten().astype(float))
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def detect_faces(image):
    """Detect faces using Haar cascade."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def predict_image(image_path, show=False):
    """Run face detection + mask prediction on an image."""
    model, scaler, metrics = load_model()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    faces = detect_faces(img)
    output = img.copy()
    results = []

    if len(faces) == 0:
        print("⚠ No faces detected in the image.")
        cv2.putText(output, "No Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Crop face with padding
            pad = int(0.1 * h)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            face_crop = img[y1:y2, x1:x2]

            # Predict
            feats = extract_features(face_crop)
            feats_s = scaler.transform(feats)
            pred    = model.predict(feats_s)[0]
            prob    = model.predict_proba(feats_s)[0]
            conf    = prob[pred]

            label   = "Mask ✓" if pred == 1 else "No Mask ✗"
            color   = (0, 255, 136) if pred == 1 else (51, 51, 255)
            results.append({'label': label, 'confidence': conf, 'bbox': (x, y, w, h)})

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Confidence
            conf_text = f"{conf*100:.1f}%"
            cv2.putText(output, conf_text, (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"  Face ({x},{y}) → {label}  [{conf*100:.1f}% confidence]")

    # Save output
    out_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(out_path, output)
    print(f"\n  Result saved: {out_path}")

    if show:
        cv2.imshow("Face Mask Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict mask on image')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--show',  action='store_true', help='Show result window')
    args = parser.parse_args()
    predict_image(args.image, args.show)
