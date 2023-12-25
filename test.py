import streamlit as st
# from PIL import Image
# from keras.models import load_model
# from util import classify, set_background
#
# # Load pneumonia classifier model
# model = load_model('keras_model.h5')
#
#
# # Load class names for pneumonia classification
# with open('labels.txt', 'r') as f:
#     class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
#
# # Set background
# set_background('bgs/01.png')
#
# st.title('Cardiac Classification')
#
# st.text("Upload an ECG image to check for Abnormal Heart Beats / Covid-19 / Normal CASE :")
#
# uploaded_file = st.file_uploader('CHOOSE A FILE', type=['jpeg', 'jpg', 'png'])
#
# if uploaded_file is not None:
#     st.write("Filename:", uploaded_file.name)
#     uploaded_image = Image.open(uploaded_file)
#     st.image(uploaded_image, caption='Input', width=200)
#
#     # Classify the image using the pneumonia classifier
#     class_name, conf_score = classify(uploaded_image, model, class_names)
#
#     st.write("## Classification Result")
#     st.write(f"ECG CLASS: {class_name}")
#     st.write(f"Confidence: {int(conf_score * 1000) / 10}%")
#

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<C:/Users/djadh/Desktop/LEARNING/NIRMAL GAUD/ECG Cardiac_ML_DL_AUTOML_XAI/ECG_Covid19_New/MI_History/PMI (49)>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
