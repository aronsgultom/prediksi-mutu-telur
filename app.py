import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ========================
# KONFIGURASI HALAMAN
# ========================
st.set_page_config(page_title="Prediksi Mutu Telur", layout="centered")
st.title("🥚 Prediksi Mutu Telur (KNN)")
st.write("Upload gambar telur untuk mengetahui kelas mutunya.")

# ========================
# LOAD MODEL
# ========================
MODEL_PATH = "model_telur.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ File model_telur.keras tidak ditemukan!")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['mutu1', 'mutu2', 'mutu3', 'mutu4']

# ========================
# FUNGSI PREPROCESS
# ========================
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# UPLOAD GAMBAR
# ========================
uploaded_file = st.file_uploader("Upload Gambar Telur", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        img_array = preprocess_image(image)

        preds = model.predict(img_array)[0]
        class_index = np.argmax(preds)
        confidence = np.max(preds) * 100

        predicted_class = CLASS_NAMES[class_index]

        st.success(f"✅ Hasil Prediksi: **{predicted_class.upper()}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        st.write("Probabilitas tiap kelas:")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"- {cls} : {preds[i]*100:.2f}%")

