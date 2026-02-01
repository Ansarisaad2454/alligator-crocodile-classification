import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐊 Alligator vs Crocodile Image Classifier")

# select model
model_name = st.selectbox(
    "Select Model",
    ("CNN", "ResNet50", "MobileNetV2", "EfficientNetB0", "VGG16")
)

# load model
@st.cache_resource
def load_model(name):
    if name == "CNN":
        return tf.keras.models.load_model("cnn_model.h5")
    elif name == "ResNet50":
        return tf.keras.models.load_model("resnet50_model.h5")
    elif name == "MobileNetV2":
        return tf.keras.models.load_model("mobilenetv2_model.h5")
    elif name == "VGG16":
        return tf.keras.models.load_model("vgg16_model.h5")
    else:
        return tf.keras.models.load_model("efficientnetb0_model.h5")

model = load_model(model_name)

# set image size based on model
if model_name == "CNN":
    img_size = (128, 128)
else:
    img_size = (224, 224)

# upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # open image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess image
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)[0][0]

    # output
    if prediction > 0.5:
        label = "Crocodile"
        confidence = prediction * 100
    else:
        label = "Alligator"
        confidence = (1 - prediction) * 100

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
