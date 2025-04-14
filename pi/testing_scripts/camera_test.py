import cv2
import time

def take_photo():
    print("Taking Photo")
    # Open webcam (0 = /dev/video0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open the camera.")
        exit(1)

    # Give camera time to adjust (especially on Raspberry Pi)
    time.sleep(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image.")
        exit(1)

    # Save the captured image
    filename = "test.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Photo saved as {filename}")

if __name__ == "__main__":
    take_photo()