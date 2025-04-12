import os
import cv2

DATASET_DIR = "rubbish-data"  # Root directory to scan

bad_files = []

for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            # Check if file is empty or unreadable
            if os.path.getsize(path) == 0:
                bad_files.append(path)
            else:
                img = cv2.imread(path)
                if img is None:
                    bad_files.append(path)

# Remove bad files
for path in bad_files:
    print(f"Removing corrupt or unreadable file: {path}")
    os.remove(path)

print(f"\n✅ Cleaned {len(bad_files)} files.")
