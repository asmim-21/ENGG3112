import random
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

# Load the Keras model (.h5)
model = tf.keras.models.load_model("rubbish_classifier.h5")

class_names = ["Non-Recycling", "Nothing", "Recycling"]  # Adjust based on your categories

def dummy_classifier(image_path):
    return random.choice(["Non-Recycling", "Nothing", "Recycling"])

def preprocess_image(image):
    """ Resize and normalize image for model input """
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

def interpret(frame):
    # Preprocess image
    input_image = preprocess_image(frame)

    # Run inference
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)

    # Display result
    label = class_names[predicted_class]
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print(f"Prediction: {label}")

    return label
