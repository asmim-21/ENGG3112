import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load model and class names
model = tf.keras.models.load_model("rubbish_classifier.h5")
class_names = ["Non-Recycling", "Nothing", "Recycling"]

# Get a batch from test_dataset
for images, labels in test_dataset.take(1):
    preds = model.predict(images)
    pred_labels = tf.argmax(preds, axis=1)

    def show_random_images():
        indices = np.random.choice(len(images), size=9, replace=False)
        fig = plt.figure(figsize=(12, 12))

        for i, idx in enumerate(indices):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(images[idx].numpy())
            true_label = labels[idx].numpy()
            predicted_label = pred_labels[idx].numpy()
            ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # Keep showing new sets of random predictions
    while True:
        show_random_images()
