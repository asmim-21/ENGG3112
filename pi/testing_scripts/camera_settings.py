import subprocess
import cv2
import time
import os
import numpy as np
from datetime import datetime
import contextlib

# Configuration
DEVICE = "/dev/video0"
CV2_INDEX = 0  # cv2.VideoCapture index
CAMERA_WARM_UP_TIME = 2  # seconds
NUM_FRAMES_TO_DISCARD = 5
OUTPUT_IMAGE = "captured_image.jpg"

# Camera settings
SETTINGS = {
    "brightness": 255,
    "contrast": 5,
    "saturation": 83,
    "white_balance_automatic": 1,
    "power_line_frequency": 2,  # 60 Hz
    "sharpness": 50,
    "backlight_compensation": 0,
    "auto_exposure": 1,  # Manual mode
    "exposure_time_absolute": 21,
    "pan_absolute": 0,
    "tilt_absolute": 0,
    "zoom_absolute": 0
}

@contextlib.contextmanager
def suppress_stderr():
    import sys
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull)

def apply_camera_settings(settings, device):
    print(f"🎛️ Applying settings to {device}...\n")
    for ctrl, val in settings.items():
        cmd = f"v4l2-ctl -d {device} -c {ctrl}={val}"
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)
            print(f"✅ {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {cmd}\n{e}")

    print("\n✅ Final settings:")
    subprocess.run(f"v4l2-ctl -d {device} --all", shell=True)

def capture_image(output_path):
    with suppress_stderr():
        cap = cv2.VideoCapture(CV2_INDEX)

    if not cap.isOpened():
        print("❌ Failed to open the camera.")
        return

    time.sleep(CAMERA_WARM_UP_TIME)

    # Discard initial unstable frames
    for _ in range(NUM_FRAMES_TO_DISCARD):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image.")
        return

    cv2.imwrite(output_path, frame)
    print(f"📸 Image saved as: {output_path}")

if __name__ == "__main__":
    apply_camera_settings(SETTINGS, DEVICE)
    time.sleep(1)
    capture_image(OUTPUT_IMAGE)
