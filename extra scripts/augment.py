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
BRIGHTNESS_VARIATION = 0.60
WHITE_BALANCE_VARIATION = 0.20
MAX_TRANSLATION = 50  # pixels
MAX_ROTATION = 7      # degrees

# --- Helper Functions ---

def adjust_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def adjust_white_balance(img, factor):
    # Simple RGB scaling
    wb_img = img.astype(np.float32)
    ch_factor = np.array([1 + factor * random.choice([-1, 1]) for _ in range(3)])
    wb_img *= ch_factor
    return np.clip(wb_img, 0, 255).astype(np.uint8)

def translate_and_rotate(img, max_translation, max_rotation_deg):
    h, w = img.shape[:2]
    tx = random.randint(-max_translation, max_translation)
    ty = random.randint(-max_translation, max_translation)
    angle = random.uniform(-max_rotation_deg, max_rotation_deg)

    # Create translation + rotation matrix
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    img = cv2.warpAffine(img, M_translation, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = cv2.warpAffine(img, M_rotation, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img

def augment_image(img):
    augments = []

    # 2 brightness variations
    for _ in range(2):
        factor = 1 + random.uniform(-BRIGHTNESS_VARIATION, BRIGHTNESS_VARIATION)
        augments.append(adjust_brightness(img, factor))

    # 2 white balance variations
    for _ in range(2):
        factor = random.uniform(-WHITE_BALANCE_VARIATION, WHITE_BALANCE_VARIATION)
        augments.append(adjust_white_balance(img, factor))

    # 2 translations + rotations
    for _ in range(2):
        augments.append(translate_and_rotate(img, MAX_TRANSLATION, MAX_ROTATION))

    # 1 original
    augments.append(img.copy())

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

print("✅ Augmentation complete! All images saved to:", OUTPUT_DIR)
