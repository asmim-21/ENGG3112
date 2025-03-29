# Rubbish Classification Model

## Project Overview
This project is a deep learning-based rubbish classification system designed to categorize waste into three categories:
- **General Waste**
- **Organics**
- **Recycling**

A **Convolutional Neural Network (CNN)** model is trained using TensorFlow/Keras, and the model can be deployed on a **Raspberry Pi** for real-time classification.

---

## 1️⃣ Setup Instructions

### **1. Set Up a Python Virtual Environment**
For better dependency management, set up a Python virtual environment:

```bash
python3 -m venv rubbish-env
source rubbish-env/bin/activate  # On Linux/Mac
rubbish-env\Scripts\activate    # On Windows
```

Once activated, install dependencies:
```bash
pip install -r requirements.txt
```

### **2. Set Up a Virtual Machine (VM) on Ubuntu**
If you are using a VM for training, follow these steps:

1. **Install VirtualBox or VMware**
   - Download and install VirtualBox or VMware Workstation Player.

2. **Create a New VM**
   - Allocate at least **4GB RAM**, **2 CPU cores**, and **20GB+ storage**.
   - Attach an Ubuntu ISO and install Ubuntu.

3. **Update and Install Dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3 python3-pip unzip -y
   ```

4. **Enable GPU Support (Optional, for Faster Training)**
   - If using an **NVIDIA GPU**, install CUDA and cuDNN:
     ```bash
     sudo apt install nvidia-driver-470
     ```

### **3. Extract the Dataset**
Download and extract the dataset:

```bash
unzip rubbish-data.zip -d dataset
```
This will create a folder structure:
```
dataset/
│── train/
│   ├── general/
│   ├── organics/
│   ├── recycling/
│── val/
│   ├── general/
│   ├── organics/
│   ├── recycling/
│── test/
│   ├── general/
│   ├── organics/
│   ├── recycling/
```

---

## 2️⃣ Training the Model

### **1. Run the Training Script**
Train the CNN model with the following command:
```bash
python train.py
```
This will:
- Load and preprocess images
- Train a CNN using TensorFlow/Keras
- Save the trained model as `rubbish_classifier.h5`

### **2. Evaluate Model Performance**
After training, test the model:
```bash
python evaluate.py
```
Expected output:
```
Test Accuracy: 85% (example output)
```

---

## 3️⃣ Deploying on Raspberry Pi

### **1. Convert Model to TensorFlow Lite (TFLite)**
Convert the trained model for Raspberry Pi:
```bash
python convert_to_tflite.py
```
This generates `rubbish_classifier.tflite`.

### **2. Set Up Raspberry Pi**
On the Raspberry Pi, install TensorFlow Lite:
```bash
pip install tflite-runtime
```
Copy `rubbish_classifier.tflite` to the Raspberry Pi.

### **3. Run Real-Time Classification**
Use a camera to classify waste:
```bash
python classify_rubbish.py
```

---

## 4️⃣ Future Improvements
- Improve accuracy with **data augmentation**.
- Optimize for Raspberry Pi with **quantization**.
- Integrate into a smart bin system.