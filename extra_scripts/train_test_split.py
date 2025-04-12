import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- Settings ---
SOURCE_DIR = "original_data"
DEST_DIR = "rubbish-data"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test
CLASSES = ["recycling", "general", "nothing"]

# --- Ensure destination folders exist ---
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        Path(f"{DEST_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# --- Perform split ---
for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIOS[0] * total)
    val_end = train_end + int(SPLIT_RATIOS[1] * total)

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_map.items():
        for file in tqdm(files, desc=f"Copying {split}/{cls}"):
            src = os.path.join(class_dir, file)
            dst = os.path.join(DEST_DIR, split, cls, file)
            shutil.copy2(src, dst)

print("✅ Dataset successfully split into train/val/test folders at:", DEST_DIR)
