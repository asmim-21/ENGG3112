import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define dataset paths
train_dir = "rubbish-data/train"
val_dir = "rubbish-data/val"
test_dir = "rubbish-data/test"

# Set image properties
IMG_SIZE = (224, 224)  # Resize to match CNN input
BATCH_SIZE = 32

# Load datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Check class names (should be ['general', 'organics', 'recycling'])
print(train_dataset.class_names)

# Normalize images (convert from [0,255] → [0,1])
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to datasets
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.2)
])

# Apply augmentation only to training data
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

from tensorflow.keras import models, layers

# Define CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse because labels are integers
              metrics=['accuracy'])

# Show model summary
model.summary()

EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}")

model.save("rubbish_classifier.h5")
print("Model saved successfully!")