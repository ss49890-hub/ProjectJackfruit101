from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import subprocess
from pydub import AudioSegment
import io
from supabase import create_client
import uuid

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model_cnn4.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASSES    = ["กล่อง", "ขวดน้ำ", "นิ้วชี้"]
SAMPLE_RATE = 16000
DURATION    = 0.3
N_MFCC      = 9
N_FFT       = 320
HOP_LENGTH  = 160

def extract_mfcc(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path.replace(".webm", ".wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_in_path,
        "-ar", "16000",
        "-ac", "1",
        tmp_out_path
    ], capture_output=True)
    y, sr = librosa.load(tmp_out_path, sr=SAMPLE_RATE, duration=DURATION)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    os.remove(tmp_in_path)
    os.remove(tmp_out_path)
    return mfcc_norm[np.newaxis, ..., np.newaxis].astype(np.float32)

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
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        result = {
            "class"      : CLASSES[np.argmax(prediction)],
            "confidence" : float(np.max(prediction) * 100),
            "all"        : {CLASSES[i]: float(prediction[i] * 100) for i in range(len(CLASSES))}
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์"}), 400
    file = request.files["file"]
    label = request.form.get("label", "unknown")
    predicted = request.form.get("predicted", "unknown")
    audio_bytes = file.read()
    filename = f"{label}/{uuid.uuid4()}.webm"
    try:
        supabase.storage.from_("audio-feedback").upload(
            filename, audio_bytes, {"content-type": "audio/webm"}
        )
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
