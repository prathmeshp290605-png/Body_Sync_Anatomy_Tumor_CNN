# Body Sync Anatomy — CNN Tumor Prediction

A B.Tech CSE educational/research prototype that uses a CNN-based image
classification model for tumor-category prediction from MRI images.

## Important
This is not a clinical diagnostic system. Predictions depend on the training
dataset and should not be used for medical decisions.

## Dataset structure

Put your images inside:

dataset/
    class_1/
        image1.jpg
        image2.jpg
    class_2/
        image1.jpg
        image2.jpg

For a common brain-tumor dataset, the folders may be:

dataset/
    glioma/
    meningioma/
    pituitary/
    notumor/

The code automatically reads the folder names as class labels.

## Run locally

### 1. Create virtual environment

Windows:

    py -3.11 -m venv venv
    venv\Scripts\activate

### 2. Install packages

    python -m pip install --upgrade pip
    pip install -r requirements.txt

### 3. Add dataset

Copy your image dataset into the `dataset` folder using one folder per class.

### 4. Train CNN

    python train.py

After training, these files appear:

    models/tumor_cnn.keras
    models/class_names.json

### 5. Start web application

    streamlit run app.py

The browser will open the Body Sync Anatomy web application.

## Project flow

MRI Image
   ↓
Resize to 224×224
   ↓
Data augmentation
   ↓
CNN feature extraction (MobileNetV2)
   ↓
Global Average Pooling
   ↓
Dense + Softmax
   ↓
Predicted tumor class + confidence

## Why MobileNetV2?

MobileNetV2 is a CNN architecture that is comparatively lightweight and
well-suited to a student web application. It is used here with transfer
learning, then the final classification layer is trained on the project
dataset.
