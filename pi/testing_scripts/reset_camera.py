import subprocess
import cv2
import time

# Set your video device
device = "/dev/video0"

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.output.decode()}")

# Step 1: Show current settings
print(f"🔍 Current settings for {device}:\n")
run_command(f"v4l2-ctl -d {device} --all")

# Step 2: Reset camera to default
print(f"\n🔄 Resetting camera settings on {device} to defaults...\n")
run_command(f"v4l2-ctl -d {device} --reset")

# Step 3: Show settings after reset
print(f"\n✅ Settings after reset for {device}:\n")
run_command(f"v4l2-ctl -d {device} --all")

# Step 4: Capture and display a frame using OpenCV
print("\n📷 Capturing image with new settings...\n")

# OpenCV needs numeric device index — /dev/video0 = 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit(1)

# Give the camera a moment to warm up
time.sleep(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture image.")
    exit(1)

# Step 5: Show the image in a file
cv2.imwrite("captured_frame.jpg", frame)
print("📷 Image saved as captured_frame.jpg")
