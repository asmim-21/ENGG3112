import pigpio
import time

SERVO_PIN = 18

pi = pigpio.pi()
if not pi.connected:
    exit()

try:
    while True:
        pi.set_servo_pulsewidth(SERVO_PIN, 1500)  # Left
        time.sleep(2)
        pi.set_servo_pulsewidth(SERVO_PIN, 2000)  # Right
        time.sleep(2)
        pi.set_servo_pulsewidth(SERVO_PIN, 1750)  # Center
        time.sleep(2)

except KeyboardInterrupt:
    pass

finally:
    pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop pulses
    pi.stop()
