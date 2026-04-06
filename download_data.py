import os
import numpy as np
import librosa
import subprocess
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
DURATION    = 0.3
N_MFCC      = 9
N_FFT       = 320
HOP_LENGTH  = 160
CLASSES     = ["raw", "ripe", "overripe"]
LABEL_MAP   = {"raw": 0, "ripe": 1, "overripe": 2}

def convert_to_wav(webm_path):
    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", webm_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], capture_output=True)
    return wav_path

def extract_mfcc(filepath):
    try:
        wav_path = convert_to_wav(filepath)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        os.remove(wav_path)
        return mfcc_norm[..., np.newaxis].astype(np.float32)
    except Exception as e:
        print(f"Error: {filepath} — {e}")
        return None

X, y = [], []
for label in CLASSES:
    folder = f"data/{label}"
    if not os.path.exists(folder):
        continue
    for filename in os.listdir(folder):
        if not filename.endswith(".webm"):
            continue
        filepath = os.path.join(folder, filename)
        mfcc = extract_mfcc(filepath)
        if mfcc is not None:
            X.append(mfcc)
            y.append(LABEL_MAP[label])
            print(f"Loaded: {label}/{filename}")

if len(X) < 10:
    print("ข้อมูลน้อยเกินไป ต้องการอย่างน้อย 10 samples")
    exit()

X = np.array(X)
y = np.array(y)
print(f"Total: {len(X)} samples, Shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model_cnn4.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved: model_cnn4.tflite")
