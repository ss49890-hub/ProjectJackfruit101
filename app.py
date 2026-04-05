from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import subprocess
from pydub import AudioSegment
import io

app = Flask(__name__)

# โหลดโมเดล tflite
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
FFMPEG_PATH = "ffmpeg"  # Render มี ffmpeg ในตัวเลย

def extract_mfcc(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32768.0
    
    target_len = int(16000 * 0.3)
    
    # หาจุดที่เสียงดังที่สุด แล้วตัด 0.3 วิรอบๆ จุดนั้น
    if len(samples) > target_len:
        energy = np.array([
            np.sum(samples[i:i+160]**2)
            for i in range(0, len(samples)-160, 160)
        ])
        peak_frame = np.argmax(energy)
        peak_sample = peak_frame * 160
        
        start = max(0, peak_sample - target_len // 2)
        end = start + target_len
        if end > len(samples):
            end = len(samples)
            start = max(0, end - target_len)
        
        samples = samples[start:end]
    
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)), mode='constant')
    
    mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=9, n_fft=320, hop_length=160)
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
