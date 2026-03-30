from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import subprocess

app = Flask(__name__)

model = tf.keras.models.load_model("model_cnn4.keras")
CLASSES = ["กล่อง", "ขวดน้ำ", "นิ้วชี้"]

SAMPLE_RATE = 16000
DURATION    = 0.3
N_MFCC      = 9
N_FFT       = 320
HOP_LENGTH  = 160

FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"

def extract_mfcc(audio_bytes):
    # บันทึกเป็น temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    # แปลง webm → wav
    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", tmp_in_path,
        "-ar", "16000",
        "-ac", "1",
        tmp_out_path
    ], capture_output=True)

    # โหลดและสกัด MFCC
    y, sr = librosa.load(tmp_out_path, sr=SAMPLE_RATE, duration=DURATION)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

    # ลบ temp file
    os.remove(tmp_in_path)
    os.remove(tmp_out_path)

    return mfcc_norm[np.newaxis, ..., np.newaxis]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์เสียง"}), 400
    file = request.files["file"]
    audio_bytes = file.read()
    try:
        mfcc = extract_mfcc(audio_bytes)
        prediction = model.predict(mfcc)[0]
        result = {
            "class"      : CLASSES[np.argmax(prediction)],
            "confidence" : float(np.max(prediction) * 100),
            "all"        : {CLASSES[i]: float(prediction[i] * 100) for i in range(len(CLASSES))}
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)