import cv2
import numpy as np
import tensorflow.lite as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="rubbish_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

def preprocess_image(image):
    """ Resize and normalize image for model input """
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

# Open webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

class_names = ["Non-Recycling", "Recycling"]  # Adjust based on your categories

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to camture image")
        break

    # Preprocess image
    input_image = preprocess_image(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    # Display result
    label = class_names[predicted_class]
    cv2.putText(frame, f"Prediction: {label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Rubbish Classification", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
