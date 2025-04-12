import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("rubbish_classifier.h5")

# Export as SavedModel for conversion
model.export("rubbish_classifier_savedmodel")

print("✅ Exported model to 'rubbish_classifier_savedmodel'")
