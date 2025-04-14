import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# --- Settings ---
INPUT_DIR = "original_data"
OUTPUT_DIR = "augmented"
CLASSES = ["recycling", "general", "nothing"]
BRIGHTNESS_VARIATION = 0.20  # ±20%
WHITE_BALANCE_VARIATION = 0.20  # ±20%

# --- Helper Functions ---

def adjust_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def adjust_white_balance(img, variation):
    img = img.astype(np.float32)
    # Apply per-channel random variation
    for c in range(3):  # R, G, B
        factor = 1 + random.uniform(-variation, variation)
        img[:, :, c] *= factor
    return np.clip(img, 0, 255).astype(np.uint8)

def augment_image(img):
    augments = []
    
    # Brightness-adjusted version
    brightness_factor = 1 + random.uniform(-BRIGHTNESS_VARIATION, BRIGHTNESS_VARIATION)
    augments.append(adjust_brightness(img, brightness_factor))
    
    # White balance-adjusted version
    augments.append(adjust_white_balance(img, WHITE_BALANCE_VARIATION))
    
    return augments

# --- Main Loop ---

for cls in CLASSES:
    input_path = os.path.join(INPUT_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(images, desc=f"Augmenting {cls}"):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Skipped unreadable image {img_path}")
            continue

        augmented_imgs = augment_image(img)

        base_name = os.path.splitext(img_name)[0]
        for i, aug_img in enumerate(augmented_imgs):
            out_name = f"{base_name}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_path, out_name), aug_img)

print("✅ Brightness and white balance augmentation complete! All images saved to:", OUTPUT_DIR)
