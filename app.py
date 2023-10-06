import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from brain_tumor.pipeline import prediction

# Set Streamlit page title and favicon
st.set_page_config(
    page_title="Brain Tumor Image Classification WebApp",
    page_icon="🥭",
)

# Load the pre-trained model
model = load_model("final_model/brain_model.h5")

# Define class labels
class_labels = {
    0: "giloma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",

}

# Define app title and header
st.title("Brain Tumor Image Classifier")
st.write("Upload an image of a brain to classify if it has a tumor or not")

# Create a file uploader widget
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Display a placeholder for the prediction result
prediction_result = st.empty()

# Check if an image has been uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Brain Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_image, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    result = model.predict(img_array)
    class_index = np.argmax(result, axis=1)

    # Determine whether the result is Healthy or Disease
    prediction = "Tumor" if class_index[0] == 5 else "No Tumor"

    # Display the prediction result in bold
    prediction_result.markdown(f"**Prediction: {prediction}**")

# Add some space and a footer
st.write("\n\n")
st.markdown(
    "Made with ❤️ by Gaurav Bhattacharya\n"
    
)