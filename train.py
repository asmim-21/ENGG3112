import os
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard
import datetime

# --- Paths and Constants ---
train_dir = "rubbish-data/train"
val_dir = "rubbish-data/val"
test_dir = "rubbish-data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# --- Load datasets ---
train_dataset = image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)

val_dataset = image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

test_dataset = image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# --- Check class names and get label order ---
class_names = train_dataset.class_names
print("Class names:", class_names)

# --- Normalize images ---
normalization_layer = layers.Rescaling(1. / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# --- Compute class weights ---
# Unbatch the dataset to extract all labels
all_labels = []
for _, label_batch in train_dataset.unbatch():
    all_labels.append(label_batch.numpy())
all_labels = np.array(all_labels)

# Compute balanced weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=all_labels
)
class_weight = dict(enumerate(class_weights_array))
print("Computed class weights:", class_weight)

# --- Define CNN model ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Setup TensorBoard log directory ---
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# --- Train model with TensorBoard and class weights ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[tensorboard_callback]
)

# --- Evaluate and save ---
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}")

model.save("rubbish_classifier.h5")
print("Model saved successfully!")


# --- Visualize random predictions ---
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Get a batch from the test dataset
for images, labels in test_dataset.take(1):
    preds = model.predict(images)
    pred_labels = tf.argmax(preds, axis=1)

    # Create a shuffled list of indices
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    # Plot 9 random samples from the batch
    plt.figure(figsize=(12, 12))
    for i in range(9):
        idx = indices[i]
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx].numpy())
        true_label = labels[idx].numpy()
        predicted_label = pred_labels[idx].numpy()
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break