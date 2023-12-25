
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
from io import StringIO
import PIL
from PIL import Image
import requests
from io import BytesIO
from keras.models import load_model
from util import classify, set_background


# Function to set background


# Load pneumonia classifier model
model = load_model('model/pneumonia_classifier.h5')

# Load YOLO model
yolo_model = YOLO("model/V8mbest.pt")

# Load class names for pneumonia classification
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Set background
set_background('bgs/01.png')

st.title('Pneumonia Detection and Classification')

st.text("Upload an image to check for pneumonia:")

uploaded_file = st.file_uploader('CHOOSE A FILE', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Input', width=200)

    # Classify the image using the pneumonia classifier
    class_name, conf_score = classify(uploaded_image, model, class_names)

    st.write("## Classification Result")
    st.write(f"Pneumonia: {class_name}")
    st.write(f"Confidence: {int(conf_score * 1000) / 10}%")

    st.divider()

    if class_name == "PNEUMONIA":
        st.text("Pneumonia detected. Performing object detection:")

        conf = st.slider('Set confidence level percentage', 0, 100, 25)

        # Perform object detection using YOLO
        confi = conf / 100
        yolo_result = yolo_model(uploaded_image, conf=confi)
        boxes = yolo_result[0].boxes

        if boxes is not None:
            img_with_boxes = yolo_result[0].plot()
            st.image(img_with_boxes, caption='Object Detection Result', width=600)


# Add a section for "Llama Medical AI Expert"
st.markdown("### LlaVA Medical AI Expert")
st.write("For more advanced medical AI expertise and consultations, visit LlaVA Medical AI Expert.")
st.write("Click [here](https://19d0-35-199-156-98.ngrok-free.app/) to access Llama Medical AI Expert.")



