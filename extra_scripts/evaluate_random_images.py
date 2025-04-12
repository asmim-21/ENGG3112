import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load model and class names
model = tf.keras.models.load_model("rubbish_classifier.h5")
class_names = ["Non-Recycling", "Nothing", "Recycling"]

# Load test dataset (adjust path and image size as needed)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "rubbish-data/test",  # Your test data folder
    image_size=(224, 224),  # Match your model input size
    batch_size=32
)

# Take one batch from test data
test_images, test_labels = next(iter(test_dataset))
preds = model.predict(test_images)
pred_labels = tf.argmax(preds, axis=1)

# Function to show 9 random predictions
def show_random_images():
    indices = np.random.choice(len(test_images), size=9, replace=False)
    fig = plt.figure(figsize=(12, 12))

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(test_images[idx].numpy().astype("uint8"))
        true_label = test_labels[idx].numpy()
        predicted_label = pred_labels[idx].numpy()
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Loop to keep showing new sets until closed
try:
    while True:
        show_random_images()
except KeyboardInterrupt:
    print("Stopped by user.")
