import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("rubbish_classifier.h5")

# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for size & performance
tflite_model = converter.convert()

# Save the converted model
with open("rubbish_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as rubbish_classifier.tflite")
