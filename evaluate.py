import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

evaluate_pretrained = False

if evaluate_pretrained:
    MODEL_PATH = "rubbish_classifier2.h5"
else:
    MODEL_PATH = "rubbish_classifier.h5"

# Load test dataset
test_dir = "rubbish-data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}")

# Get predictions
y_true = []
y_pred = []
class_names = test_dataset.class_names

for images, labels in test_dataset:
    # --- Predict ---
    if evaluate_pretrained:
        # Apply MobileNetV2 preprocessing
        images = preprocess_input(images)
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=class_names))
