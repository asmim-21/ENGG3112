import os
from collections import Counter

def count_images(folder):
    count = Counter()
    for cls in os.listdir(folder):
        cls_folder = os.path.join(folder, cls)
        if os.path.isdir(cls_folder):
            count[cls] = len([
                f for f in os.listdir(cls_folder)
                if f.lower().endswith(('jpg', 'jpeg', 'png'))
            ])
    return count

print("Train class counts:", count_images("rubbish-data/train"))
