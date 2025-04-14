import board
import neopixel
import time

# ----- Constants -----
LED_PIN = board.D18            # GPIO18 (PWM0)
NUM_LEDS = 5                   # Number of LEDs in the strip
LED_BRIGHTNESS = 0.5           # Between 0.0 (off) and 1.0 (max)
LED_COLOR = (255, 255, 180)    # RGB for white

# ----- Setup -----
pixels = neopixel.NeoPixel(
    LED_PIN,
    NUM_LEDS,
    brightness=LED_BRIGHTNESS,
    auto_write=True,
    pixel_order=neopixel.GRB  # Most WS2812B strips use GRB
)

# ----- Turn all LEDs on white -----
for i in range(NUM_LEDS):
    pixels[i] = LED_COLOR