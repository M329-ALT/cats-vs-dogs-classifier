import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

def preprocess_image(image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

st.title("🐱🐶 تصنيف القطط والكلاب")
uploaded_file = st.file_uploader("ارفع صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة التي رفعتها", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    
    if prediction > 0.5:
        result = "كلب 🐶"
        confidence = prediction
    else:
        result = "قط 🐱"
        confidence = 1 - prediction

    st.success(f"النتيجة: {result}")
    st.info(f"نسبة الثقة: {confidence:.2%}")
