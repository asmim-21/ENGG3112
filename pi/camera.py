import cv2
import time
import os
import contextlib
import numpy as np
import subprocess
from datetime import datetime

CAMERA_WARM_UP_TIME = 2  # seconds
IMAGE_DIR = "images"
NUM_FRAMES_TO_DISCARD = 5
DEVICE = "/dev/video0"

# Camera settings (adjust as needed)
SETTINGS = {
    "brightness": 30,
    "contrast": 5,
    "saturation": 83,
    "white_balance_automatic": 1,
    "power_line_frequency": 2,  # 2 = 60 Hz
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
    for ctrl, val in settings.items():
        cmd = f"v4l2-ctl -d {device} -c {ctrl}={val}"
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to apply setting: {ctrl} = {val}\n{e}")


def simulate_lower_exposure(frame, gamma=1.8):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def capture_image(cap):
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return None

    # Discard initial unstable frames
    for _ in range(NUM_FRAMES_TO_DISCARD):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to capture image.")
        return None

    os.makedirs(IMAGE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{IMAGE_DIR}/capture_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Image saved: {filename}")

    return frame

def camera_init():
    print("Initialising Camera...")
    apply_camera_settings(SETTINGS, DEVICE)

    with suppress_stderr():
        cap = cv2.VideoCapture(0)

    # ✅ Set resolution ONCE here
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    time.sleep(CAMERA_WARM_UP_TIME)

    print("Done")

    return cap


if __name__ == "__main__":
    capture_image()
