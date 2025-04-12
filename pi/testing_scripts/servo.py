# servo.py
import time

# Pulse widths (in microseconds)
LEFT = 2000
CENTER = 1630
RIGHT = 1320

def center_servo(pi, pin, move_time, tip_time):
    pi.set_servo_pulsewidth(pin, CENTER)
    time.sleep(move_time)
    pi.set_servo_pulsewidth(pin, 0)
    time.sleep(tip_time)

def move_servo(pi, pin, direction, move_time, tip_time):
    if direction == "left":
        pi.set_servo_pulsewidth(pin, LEFT)
    elif direction == "right":
        pi.set_servo_pulsewidth(pin, RIGHT)
    else:
        return

    time.sleep(move_time)
    pi.set_servo_pulsewidth(pin, 0)
    time.sleep(tip_time)
    center_servo(pi, pin, move_time, 0)
