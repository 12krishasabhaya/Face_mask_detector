"""
Face Mask Detector — Synthetic Dataset Generator
==================================================
Generates realistic synthetic face images for training.
Run: python src/generate_dataset.py --count 500
"""

import numpy as np
import cv2
import os
import random
import argparse


def generate_face(size=128, has_mask=False, seed=None):
    """Generate a synthetic face image."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = np.ones((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = random.randint(160, 230)
    img[:, :, 1] = random.randint(170, 230)
    img[:, :, 2] = random.randint(185, 240)

    cx, cy = size // 2, size // 2
    skin = (
        random.randint(100, 210),
        random.randint(70,  160),
        random.randint(55,  130)
    )

    # Face oval
    cv2.ellipse(img, (cx, cy), (42, 52), 0, 0, 360, skin, -1)

    # Skin texture noise
    noise = np.random.randint(-12, 12, (104, 84, 3), dtype=np.int16)
    r1, r2 = cy-52, cy+52
    c1, c2 = cx-42, cx+42
    region = img[r1:r2, c1:c2].astype(np.int16)
    img[r1:r2, c1:c2] = np.clip(region + noise, 0, 255).astype(np.uint8)

    # Eyes
    for ex in [cx - 16, cx + 16]:
        cv2.ellipse(img, (ex, cy - 10), (11, 7), 0, 0, 360, (255, 255, 255), -1)
        eye_col = (random.randint(20, 80),) * 3
        cv2.circle(img, (ex, cy - 10), 5, eye_col, -1)
        cv2.circle(img, (ex - 1, cy - 11), 2, (240, 240, 240), -1)

    # Eyebrows
    br = tuple(max(0, c - 45) for c in skin)
    for ex in [cx - 16, cx + 16]:
        cv2.ellipse(img, (ex, cy - 22), (13, 4), 0, 0, 360, br, 2)

    # Ears
    cv2.ellipse(img, (cx - 44, cy), (6, 11), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx + 44, cy), (6, 11), 0, 0, 360, skin, -1)

    # Hair
    hair_col = (
        random.randint(10, 110),
        random.randint(5,  60),
        random.randint(5,  40)
    )
    cv2.ellipse(img, (cx, cy - 36), (47, 32), 0, 180, 360, hair_col, -1)
    cv2.rectangle(img, (cx - 45, cy - 52), (cx + 45, cy - 34), hair_col, -1)

    if has_mask:
        # Surgical mask
        colors = [
            (210, 210, 210),  # white
            (90,  148, 220),  # blue
            (180, 220, 180),  # green
            (220, 170, 190),  # pink
            (200, 200, 170),  # beige
        ]
        mc = random.choice(colors)
        pts = np.array([
            [cx - 37, cy + 0],
            [cx + 37, cy + 0],
            [cx + 40, cy + 32],
            [cx,      cy + 45],
            [cx - 40, cy + 32]
        ], np.int32)
        cv2.fillPoly(img, [pts], mc)
        sc = tuple(max(0, c - 25) for c in mc)
        # Horizontal pleats
        for yo in [6, 14, 22, 30]:
            cv2.line(img, (cx-36, cy+yo), (cx+36, cy+yo), sc, 1)
        # Nose wire
        cv2.line(img, (cx - 22, cy + 4), (cx + 22, cy + 4), (160, 160, 160), 2)
        # Ear loops
        cv2.line(img, (cx - 37, cy + 0), (cx - 47, cy - 14), sc, 2)
        cv2.line(img, (cx + 37, cy + 0), (cx + 47, cy - 14), sc, 2)
    else:
        # Nose
        nc = tuple(max(0, c - 22) for c in skin)
        cv2.ellipse(img, (cx, cy + 10), (8, 6), 0, 0, 360, nc, -1)
        cv2.circle(img, (cx - 6, cy + 12), 3, tuple(max(0, c - 32) for c in skin), -1)
        cv2.circle(img, (cx + 6, cy + 12), 3, tuple(max(0, c - 32) for c in skin), -1)
        # Lips
        lc = (random.randint(155, 200), random.randint(80, 120), random.randint(75, 115))
        cv2.ellipse(img, (cx, cy + 27), (20, 9), 0, 0, 180, lc, -1)
        cv2.ellipse(img, (cx, cy + 24), (20, 7), 0, 180, 360,
                    tuple(max(0, c - 25) for c in lc), -1)

    # Neck
    cv2.rectangle(img, (cx - 12, cy + 52), (cx + 12, size), skin, -1)

    # Augmentation: slight rotation
    angle = random.uniform(-18, 18)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (size, size), borderValue=img[0, 0].tolist())

    # Brightness variation
    b = random.randint(-35, 35)
    img = np.clip(img.astype(np.int16) + b, 0, 255).astype(np.uint8)

    # Optional Gaussian blur (simulate lower camera quality)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_dataset(output_dir='dataset', count=500, img_size=128):
    """Generate full dataset."""
    for label, has_mask in [('with_mask', True), ('without_mask', False)]:
        folder = os.path.join(output_dir, label)
        os.makedirs(folder, exist_ok=True)
        existing = len(os.listdir(folder))
        print(f"Generating {count} '{label}' images...")
        for i in range(count):
            img = generate_face(img_size, has_mask, seed=i + (0 if has_mask else 10000))
            path = os.path.join(folder, f'face_{existing+i:05d}.jpg')
            cv2.imwrite(path, img)
        print(f"  → {os.path.join(folder)} ({count} images)")
    print(f"\n✅ Dataset ready in '{output_dir}/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic face dataset')
    parser.add_argument('--count',  type=int, default=500, help='Images per class')
    parser.add_argument('--output', default='dataset', help='Output directory')
    parser.add_argument('--size',   type=int, default=128, help='Image size')
    args = parser.parse_args()
    generate_dataset(args.output, args.count, args.size)
