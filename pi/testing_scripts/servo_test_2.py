import pigpio
import evdev
from evdev import InputDevice, ecodes
import time
from servo import LEFT, CENTER, RIGHT

SERVO_PIN = 18

# Init pigpio
pi = pigpio.pi()
if not pi.connected:
    print("Failed to connect to pigpio daemon.")
    exit()

# Find your keyboard device
devices = [InputDevice(path) for path in evdev.list_devices()]
keyboard = None
for device in devices:
    if 'keyboard' in device.name.lower() or 'kbd' in device.name.lower():
        keyboard = InputDevice(device.path)
        break

if not keyboard:
    print("Keyboard device not found.")
    pi.stop()
    exit()

print(f"Using input device: {keyboard.name} ({keyboard.path})")
print("Press ← or → to control the servo. Release to center. Ctrl+C to stop.")

# Enable non-blocking event reading
keyboard.grab()

try:
    for event in keyboard.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = evdev.categorize(event)
            key_code = key_event.scancode
            key_state = key_event.keystate  # 0 = up, 1 = down, 2 = hold

            if key_code == ecodes.KEY_LEFT:
                if key_state == 1:
                    print("← Pressed")
                    pi.set_servo_pulsewidth(SERVO_PIN, LEFT)
                elif key_state == 0:
                    print("← Released")
                    pi.set_servo_pulsewidth(SERVO_PIN, CENTER)

            elif key_code == ecodes.KEY_RIGHT:
                if key_state == 1:
                    print("→ Pressed")
                    pi.set_servo_pulsewidth(SERVO_PIN, RIGHT)
                elif key_state == 0:
                    print("→ Released")
                    pi.set_servo_pulsewidth(SERVO_PIN, CENTER)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    keyboard.ungrab()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
