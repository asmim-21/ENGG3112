import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# --- Configuration ---
IMG_SIZE = (224, 224)
MODEL_PATH = "rubbish_classifier.h5"
IMAGE_PATH = "capture_test.jpg"  # Replace with your image filename

# --- Load the trained model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Class names (ensure consistent order with training) ---
class_names = ["general", "nothing", "recycling"]  # Adjust based on your training folder names

# --- Load and preprocess the image ---
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize (same as training)

# --- Predict ---
preds = model.predict(img_array)
pred_class_index = np.argmax(preds[0])
pred_class = class_names[pred_class_index]
confidence = preds[0][pred_class_index]

# --- Output ---
print(f"🧠 Predicted class: {pred_class} (Confidence: {confidence:.2f})")

# --- Show image with prediction ---
plt.imshow(img)
plt.title(f"Prediction: {pred_class} ({confidence:.2f})")
plt.axis("off")
plt.show()
