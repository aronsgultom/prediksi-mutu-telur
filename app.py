import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Prediksi Mutu Telur", layout="centered")
st.title("🥚 Prediksi Mutu Telur dengan CNN")

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "model_telur.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model_telur.keras tidak ditemukan!")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Ambil ukuran input model otomatis
INPUT_HEIGHT = model.input_shape[1]
INPUT_WIDTH  = model.input_shape[2]
INPUT_CHANNEL = model.input_shape[3]

st.success(f"✅ Model loaded dengan input")

# HARUS SAMA SAAT TRAINING
CLASS_NAMES = ['mutu1', 'mutu2', 'mutu3', 'mutu4']

# ==============================
# PREPROCESS GAMBAR (BEBAS UKURAN)
# ==============================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))

    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==============================
# UPLOAD GAMBAR
# ==============================
uploaded_file = st.file_uploader(
    "📤 Upload gambar telur",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    img_array = preprocess_image(img)

    if st.button("🔍 Prediksi Mutu"):
        with st.spinner("Sedang memproses..."):
            preds = model.predict(img_array, verbose=0)[0]
            class_index = np.argmax(preds)
            confidence = float(np.max(preds) * 100)

            predicted_class = CLASS_NAMES[class_index]

        st.success(f"✅ Hasil Prediksi: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        st.subheader("Probabilitas Tiap Kelas:")
        for i, c in enumerate(CLASS_NAMES):
            st.write(f"{c}: {preds[i]*100:.2f}%")

