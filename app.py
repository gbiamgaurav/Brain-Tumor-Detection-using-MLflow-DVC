import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from brain_tumor.pipeline.prediction import braindata

# Set Streamlit page title and favicon
st.set_page_config(
    page_title="Brain Tumor Image Classifier Webapp",
    page_icon="🧠",
)

# Define app title and header
st.title("Brain Tumor Image Classification")
st.write("Upload an image of a brain to classify if it has a Tumor or Not")

# Create a file uploader widget
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Display a placeholder for the prediction result
prediction_result = st.empty()

# Check if an image has been uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Brain Image", use_column_width=True)

    # Create an instance of the braindata class
    predictor = braindata(filename=uploaded_image)

    # Make a prediction using the predictor
    prediction_result_data = predictor.predictbraindata()

    # Display the prediction result in Streamlit
    st.write("Prediction:", prediction_result_data[0]['image'])

# Add some space and a footer
st.write("\n\n")
