"""
Face Mask Detector — Training Script
=====================================
Run: python src/train.py
"""

import numpy as np
import cv2
import os
import pickle
import json
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
IMG_SIZE      = 64
FEATURE_SIZE  = 32
TEST_SPLIT    = 0.2
RANDOM_STATE  = 42

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(img_path):
    """Extract HOG-like gradient + color histogram features from image."""
    img  = cv2.imread(img_path)
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE))

    # Gradient features
    gx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    feats = []
    for r in range(0, FEATURE_SIZE, 8):
        for c in range(0, FEATURE_SIZE, 8):
            bm = mag[r:r+8, c:c+8].flatten()
            ba = ang[r:r+8, c:c+8].flatten()
            feats += [bm.mean(), bm.std(), ba.mean(), ba.std()]
            hist, _ = np.histogram(ba, bins=8, range=(-np.pi, np.pi), weights=bm)
            feats += list(hist)

    # Color histograms (16 bins per channel)
    for ch in range(3):
        h = cv2.calcHist([img], [ch], None, [16], [0, 256]).flatten()
        feats += list(h)

    # Raw pixel values
    feats += list(small.flatten().astype(float))
    return feats


def load_dataset(dataset_dir=DATASET_DIR):
    """Load all images and extract features."""
    classes = [('with_mask', 1), ('without_mask', 0)]
    X, y = [], []
    for label, cls in classes:
        folder = os.path.join(dataset_dir, label)
        if not os.path.exists(folder):
            print(f"  [!] Folder not found: {folder}")
            continue
        files = sorted(os.listdir(folder))
        print(f"  Loading {len(files)} images from '{label}'...")
        for fname in files:
            path = os.path.join(folder, fname)
            feats = extract_features(path)
            X.append(feats)
            y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)


# ─── Training ─────────────────────────────────────────────────────────────────
def train(dataset_dir=DATASET_DIR):
    print("\n" + "="*50)
    print("  FACE MASK DETECTOR — Training Pipeline")
    print("="*50)

    print("\n[1/4] Loading dataset...")
    X, y = load_dataset(dataset_dir)
    print(f"  Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  With mask: {(y==1).sum()}, Without mask: {(y==0).sum()}")

    print("\n[2/4] Splitting & scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n[3/4] Training MLP Neural Network (256 → 128 → 64)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=150,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        verbose=False
    )
    model.fit(X_train_s, y_train)
    print(f"  Converged in {model.n_iter_} iterations")

    print("\n[4/4] Evaluating...")
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy:  {acc*100:.2f}%")
    print("\n" + classification_report(y_test, y_pred, target_names=['No Mask', 'Mask']))

    cv = cross_val_score(model, X_train_s, y_train, cv=5)
    print(f"  5-Fold CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    # Save artefacts
    metrics = {
        'accuracy':    round(float(acc), 4),
        'cv_mean':     round(float(cv.mean()), 4),
        'cv_std':      round(float(cv.std()), 4),
        'train_size':  int(len(X_train)),
        'test_size':   int(len(X_test)),
        'feature_dim': int(X.shape[1]),
        'model_type':  'MLPClassifier (256-128-64)',
        'iterations':  int(model.n_iter_)
    }

    with open(os.path.join(MODEL_DIR, 'mask_detector.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'metrics': metrics}, f)

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _save_plots(model, y_test, y_pred)

    print("\n✅ Training complete!")
    print(f"   Model   → {MODEL_DIR}/mask_detector.pkl")
    print(f"   Metrics → {MODEL_DIR}/metrics.json")
    print(f"   Plots   → {RESULTS_DIR}/training_results.png")
    return model, scaler, metrics


def _save_plots(model, y_test, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0f')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['No Mask', 'Mask'], yticklabels=['No Mask', 'Mask'],
                ax=axes[0], linewidths=0.5, cbar=False,
                annot_kws={'color': 'white', 'size': 14})
    axes[0].set_title('Confusion Matrix', color='white', fontsize=13)
    axes[0].set_xlabel('Predicted', color='#aaa')
    axes[0].set_ylabel('Actual', color='#aaa')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#111118')

    # Loss curve
    axes[1].plot(model.loss_curve_, color='#00ff88', linewidth=2)
    axes[1].set_title('Training Loss', color='white', fontsize=13)
    axes[1].set_xlabel('Iterations', color='#aaa')
    axes[1].set_ylabel('Loss', color='#aaa')
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#111118')

    # Dataset bar
    bars = axes[2].bar(['With Mask', 'No Mask'], [500, 500],
                       color=['#00ff88', '#ff3355'], width=0.5, edgecolor='none')
    axes[2].set_title('Dataset Distribution', color='white', fontsize=13)
    axes[2].set_ylabel('Samples', color='#aaa')
    axes[2].tick_params(colors='white')
    axes[2].set_facecolor('#111118')
    for b in bars:
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+5,
                     str(int(b.get_height())), ha='center', color='white', fontsize=12)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(RESULTS_DIR, 'training_results.png'),
                dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Face Mask Detector')
    parser.add_argument('--dataset', default=DATASET_DIR, help='Path to dataset folder')
    args = parser.parse_args()
    train(args.dataset)
