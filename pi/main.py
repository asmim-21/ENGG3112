# main.py
from setup import setup_environment, get_pi
from camera import capture_image, camera_init
from servo import center_servo, move_servo
from evaluation import dummy_classifier, interpret
import time

SERVO_PIN = 13

# Timing Constants
MOVE_TIME = 0.2  # seconds
TIP_TIME = 0.5     # seconds

if __name__ == "__main__":
    setup_environment()
    pi = get_pi()

    cap = camera_init()

    try:
        while True:
            print("Centering...")
            center_servo(pi, SERVO_PIN, MOVE_TIME, TIP_TIME)

            print("Capturing image...")
            frame = capture_image(cap)

            if frame is not None:
                result = interpret(frame)
                # result = dummy_classifier(frame)
                
                print(f"Classifier result: {result}")

                if result == "Non-Recycling":
                    print("Turning left...")
                    move_servo(pi, SERVO_PIN, direction="left", move_time=MOVE_TIME, tip_time=TIP_TIME)

                elif result == "Recycling":
                    print("Turning right...")
                    move_servo(pi, SERVO_PIN, direction="right", move_time=MOVE_TIME, tip_time=TIP_TIME)

                else:
                    print("Nothing detected. Staying centered.")
            else:
                print("Failed to capture image.")

    except KeyboardInterrupt:
        print("Program stopped by user.")

    finally:
        center_servo(pi, SERVO_PIN, MOVE_TIME, TIP_TIME)
        cap.release()
        pi.set_servo_pulsewidth(SERVO_PIN, 0)
        pi.stop()

