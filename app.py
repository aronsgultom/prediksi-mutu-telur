from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# ==============================
# KONFIGURASI FLASK
# ==============================
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# LOAD MODEL CNN (.keras)
# ==============================
MODEL_PATH = 'model_telur.keras'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model_telur.keras tidak ditemukan di folder project!")

model = tf.keras.models.load_model(MODEL_PATH)

# Ambil ukuran input langsung dari model
INPUT_HEIGHT = model.input_shape[1]
INPUT_WIDTH  = model.input_shape[2]
INPUT_CHANNEL = model.input_shape[3]

print("✅ Model input size:", model.input_shape)

# HARUS SAMA URUTAN SAAT TRAINING
CLASS_NAMES = ['mutu1', 'mutu2', 'mutu3', 'mutu4']

# ==============================
# PREPROCESS GAMBAR (BEBAS UKURAN)
# ==============================
def preprocess_image(image_path):
    # Pastikan gambar RGB (aman untuk PNG, grayscale, dll)
    img = Image.open(image_path).convert("RGB")

    # Resize sesuai kebutuhan model
    img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))

    # Konversi ke array
    img_array = np.array(img)

    # Normalisasi
    img_array = img_array.astype("float32") / 255.0

    # Tambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==============================
# PREDIKSI
# ==============================
def predict_image(image_path):
    img_array = preprocess_image(image_path)

    preds = model.predict(img_array, verbose=0)[0]
    class_index = np.argmax(preds)
    confidence = float(np.max(preds) * 100)

    predicted_class = CLASS_NAMES[class_index]

    return predicted_class, confidence

# ==============================
# ROUTING
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('predict.html', error="⚠️ Tidak ada file yang diunggah.")

        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error="⚠️ File belum dipilih.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            predicted_class, confidence = predict_image(file_path)
        except Exception as e:
            return render_template(
                'predict.html',
                error=f"❌ Terjadi kesalahan saat prediksi: {str(e)}"
            )

        return render_template(
            'predict.html',
            result=predicted_class,
            confidence=f"{confidence:.2f} %",
            filename=filename
        )

    return render_template('predict.html')

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
