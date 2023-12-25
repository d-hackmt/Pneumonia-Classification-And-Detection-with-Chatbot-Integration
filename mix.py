import streamlit as st
st.set_page_config(layout="wide")
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
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

# Function to get the response back from LLM Chatbot
def getLLMResponse(query):
    llm = CTransformers(model="meditron-7b.Q2_K.gguf",
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    template = """
    {query}
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    response = llm(prompt.format(query=query))
    return response

# Load pneumonia classifier model
model = load_model('model/pneumonia_classifier.h5')

# Load YOLO model
yolo_model = YOLO("model/V8mbest.pt")

# Load class names for pneumonia classification
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Set background
background_image = 'bgs/01.png'

set_background(background_image)

col1, col2 = st.columns(2)

# Left column for Chatbot
with col1:
    st.title("LLM Chatbot 🤖")
    st.text("ASK ME ANYTHING:")
    query = st.text_area('ENTER YOUR QUERY BELOW', height=70)
    submit = st.button("Ask")
    if submit:
        response = getLLMResponse(query)
        st.write("LLM Response:")
        st.write(response)

    st.markdown("### LlaVA Medical AI Expert 🤖 ")
    st.write("For more advanced medical AI expertise and consultations, visit LlaVA Medical AI Expert.")
    st.write("Click [here](https://308c-34-132-236-108.ngrok-free.app/) to access Llama Medical AI Expert.")

# Right column for Pneumonia Detection and Classification
with col2:
    st.title('Pneumonia Detection and Classification')
    st.text("Upload an image to check for pneumonia:")
    uploaded_file = st.file_uploader('CHOOSE A FILE', type=['jpeg', 'jpg', 'png'])

    if uploaded_file is not None:
        st.write("Filename:", uploaded_file.name)
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption='Input', width=200)

        class_name, conf_score = classify(uploaded_image, model, class_names)

        st.write("## Classification Result")
        st.write(f"Pneumonia: {class_name}")
        st.write(f"Confidence: {int(conf_score * 1000) / 10}%")

        st.divider()

        if class_name == "PNEUMONIA":
            st.text("Pneumonia detected. Performing object detection:")
            conf = st.slider('Set confidence level percentage', 0, 100, 25)
            confi = conf / 100
            yolo_result = yolo_model(uploaded_image, conf=confi)
            boxes = yolo_result[0].boxes

            if boxes is not None:
                img_with_boxes = yolo_result[0].plot()
                st.image(img_with_boxes, caption='Object Detection Result', width=600)

    # Add a section for "Llama Medical AI Expert"

