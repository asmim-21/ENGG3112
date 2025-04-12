# setup.py
import pigpio
import os

IMAGE_DIR = "images"

def setup_environment():
    os.makedirs(IMAGE_DIR, exist_ok=True)

def get_pi():
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio daemon.")
        exit()
    return pi
