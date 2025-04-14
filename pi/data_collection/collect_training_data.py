import cv2
import os
import time
from datetime import datetime
import contextlib
import subprocess

IMAGE_BASE_DIR = "original_data"
RESOLUTION = (320, 240)
CAMERA_INDEX = 0

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

def apply_camera_settings(settings, device):
    for ctrl, val in settings.items():
        cmd = f"v4l2-ctl -d {device} -c {ctrl}={val}"
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to apply setting: {ctrl} = {val}\n{e}")


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

def get_next_index(folder, object_name):
    """Get the next available index for a given object based on existing files."""
    if not os.path.exists(folder):
        return 0

    files = sorted(os.listdir(folder))
    max_index = 0
    for f in files:
        if f.startswith(object_name + "_") and f.endswith(".jpg"):
            try:
                index = int(f[len(object_name)+1:-4])
                max_index = max(max_index, index + 1)
            except ValueError:
                continue
    return max_index

def capture_image(cap, class_label, object_name, start_index):
    folder = os.path.join(IMAGE_BASE_DIR, class_label)
    os.makedirs(folder, exist_ok=True)

    index = get_next_index(folder, object_name) if start_index is None else start_index
    filename = f"{object_name}_{index}.jpg"
    filepath = os.path.join(folder, filename)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(filepath, frame)
        print(f"[{index}] Saved: {filepath}")
        return index + 1  # next index
    else:
        print("❌ Failed to capture image.")
        return index

def main():
    print("Enter class [recycling, general, nothing]:")
    class_label = input("Class: ").strip().lower()

    if class_label not in ["recycling", "general", "nothing"]:
        print("❌ Invalid class. Exiting.")
        return

    if class_label == "nothing":
        object_name = "nothing"
    else:
        object_name = input("Object name (e.g., 'plastic_bottle'): ").strip().lower().replace(" ", "_")

    folder = os.path.join(IMAGE_BASE_DIR, class_label)
    current_index = get_next_index(folder, object_name)

    print("📷 Warming up camera...")

    apply_camera_settings(SETTINGS, DEVICE)

    with suppress_stderr():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    time.sleep(1)  # Warm-up delay

    print(f"📁 Ready to capture for '{object_name}' under class '{class_label}'. Starting from index {current_index}.")
    print("🔘 Press ENTER to capture an image. Press Ctrl+C to stop.")

    try:
        while True:
            input()
            current_index = capture_image(cap, class_label, object_name, current_index)
    except KeyboardInterrupt:
        print("\n⏹️ Capture stopped.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()