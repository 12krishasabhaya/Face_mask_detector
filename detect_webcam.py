"""
Face Mask Detector — Real-Time Webcam Detection
=================================================
Run: python src/detect_webcam.py
Press Q to quit.
"""

import numpy as np
import cv2
import pickle
import os
import time


MODEL_PATH = "models/mask_detector.pkl"
IMG_SIZE   = 64
FEAT_SIZE  = 32


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


def extract_features(img):
    img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (FEAT_SIZE, FEAT_SIZE))
    gx    = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag   = np.sqrt(gx**2 + gy**2)
    ang   = np.arctan2(gy, gx)
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


def run_webcam():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found. Run: python src/train.py")
        return

    print("Loading model...")
    model, scaler = load_model()

    cascade_path  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ Webcam started! Press Q to quit.\n")

    # Stats
    stats = {'mask': 0, 'no_mask': 0, 'frames': 0}
    fps_counter, fps_time = 0, time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        stats['frames'] += 1
        fps_counter     += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time    = now

        for (x, y, w, h) in faces:
            pad = int(0.1 * h)
            x1 = max(0, x-pad);           y1 = max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad); y2 = min(frame.shape[0], y+h+pad)
            face_crop = frame[y1:y2, x1:x2]

            feats  = extract_features(face_crop)
            feats_s = scaler.transform(feats)
            pred   = model.predict(feats_s)[0]
            prob   = model.predict_proba(feats_s)[0]
            conf   = prob[pred]

            has_mask = (pred == 1)
            label    = "MASK" if has_mask else "NO MASK"
            color    = (0, 255, 136) if has_mask else (51, 51, 255)

            if has_mask: stats['mask']    += 1
            else:        stats['no_mask'] += 1

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Corners
            cs = 16
            for (bx, by, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (bx, by), (bx+dx*cs, by), color, 3)
                cv2.line(frame, (bx, by), (bx, by+dy*cs), color, 3)

            # Label pill
            tag = f"{label}  {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+12, y1), color, -1)
            cv2.putText(frame, tag, (x1+6, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

        # HUD overlay
        h_img = frame.shape[0]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 110), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        def hud(text, x, y, col=(200, 200, 200)):
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        hud(f"FPS:       {fps}", 10, 25)
        hud(f"Faces:     {len(faces)}", 10, 47)
        hud(f"Mask:      {stats['mask']}", 10, 69, (0, 255, 136))
        hud(f"No Mask:   {stats['no_mask']}", 10, 91, (51, 51, 255))

        total = stats['mask'] + stats['no_mask']
        if total > 0:
            rate = int(stats['mask'] / total * 100)
            col  = (0, 255, 136) if rate >= 70 else (0, 165, 255)
            cv2.putText(frame, f"Compliance: {rate}%", (10, h_img-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        cv2.imshow("Face Mask Detector  |  Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Session Stats:")
    print(f"   Frames processed : {stats['frames']}")
    print(f"   Mask detections  : {stats['mask']}")
    print(f"   No mask          : {stats['no_mask']}")


if __name__ == '__main__':
    run_webcam()
