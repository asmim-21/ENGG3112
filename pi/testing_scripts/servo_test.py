import RPi.GPIO as GPIO
import time

PWM_PIN = 18          # GPIO18 (Pin 12)
PWM_FREQ = 50         # 50 Hz (20 ms period)

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWM_PIN, GPIO.OUT)

# Create PWM instance with 50Hz
pwm = GPIO.PWM(PWM_PIN, PWM_FREQ)
pwm.start(5)  # Start with 1ms (5% duty cycle)

right = 2.4
left = 4.9
center = (right + left) / 2

try:
    while True:
        pwm.ChangeDutyCycle(center) 
        time.sleep(2)
        pwm.ChangeDutyCycle(left)
        time.sleep(1)
        pwm.ChangeDutyCycle(center) 
        time.sleep(2)
        pwm.ChangeDutyCycle(right)
        time.sleep(1)

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()
