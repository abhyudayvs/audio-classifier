import os
import tempfile
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Labels ──
try:
    with open("labels.txt", "r") as f:
        LABELS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    LABELS = ["Church_Bell", "Cicada", "Clapping", "Dog"] # Updated true order!
    
# ── Load Model ──
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
expected_shape = input_details[0]['shape']
target_size = np.prod(expected_shape)
model_dtype = input_details[0]['dtype']

print(f"--- AI MODEL LOADED ---")
print(f"Expected Input Shape: {expected_shape}")
print(f"Model Data Type: {model_dtype}")

def _normalize_mfcc(mfccs: np.ndarray, window_size: int = 101) -> np.ndarray:
    normalized = np.copy(mfccs)
    half = window_size // 2
    num_frames = mfccs.shape[1]
    for t in range(num_frames):
        start = max(0, t - half)
        end = min(num_frames, t + half + 1)
        normalized[:, t] = mfccs[:, t] - np.mean(mfccs[:, start:end], axis=1)
    return normalized

def extract_features(audio_window: np.ndarray, sr: int) -> np.ndarray:
    # 1. Pad to 1 second
    if len(audio_window) < 16000:
        audio_window = np.pad(audio_window, (0, 16000 - len(audio_window)), mode='constant')

    # 2. Pre-emphasis
    audio_window = librosa.effects.preemphasis(audio_window, coef=0.98)

    # 3. Exact EI Parameters
    mfccs = librosa.feature.mfcc(
        y=audio_window,
        sr=sr,
        n_mfcc=13,
        n_fft=256,
        hop_length=320,  
        win_length=256,
        n_mels=32,
        fmin=0,
        fmax=None
    )

    # 4. Normalization & Flattening
    mfccs = _normalize_mfcc(mfccs, window_size=101)
    mfccs = mfccs.T
    mfccs_flat = mfccs.flatten()

    if len(mfccs_flat) > target_size:
        mfccs_flat = mfccs_flat[:target_size]
    elif len(mfccs_flat) < target_size:
        mfccs_flat = np.pad(mfccs_flat, (0, target_size - len(mfccs_flat)), mode='constant')

    input_tensor = mfccs_flat.reshape(expected_shape)

    # 🚨 THE FINAL BOSS: EXACT TFLITE QUANTIZATION 🚨
    # If the model is Quantized (Int8 or Uint8), we MUST scale the floats using its secret formula!
    if model_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        if scale > 0.0:
            input_tensor = np.round((input_tensor / scale) + zero_point)
            input_tensor = np.clip(input_tensor, -128, 127)
            
    elif model_dtype == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        if scale > 0.0:
            input_tensor = np.round((input_tensor / scale) + zero_point)
            input_tensor = np.clip(input_tensor, 0, 255)

    return input_tensor.astype(model_dtype)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        y, sr = librosa.load(tmp_file_path, sr=16000, mono=True)
        y = y * 32768.0  # Scale to standard 16-bit PCM volume

        window_size = 16000   
        step_size = 8000    

        counts = {label: 0 for label in LABELS}
        best_confidence = {label: 0.0 for label in LABELS}
        last_detected_class = None
        cooldown = 0

        for start in range(0, len(y), step_size):
            window = y[start:start + window_size]
            if len(window) < 1600:
                continue

            # Silence Gate
            if np.max(np.abs(window)) < 250:
                last_detected_class = None
                if cooldown > 0: cooldown -= 1
                continue

            input_tensor = extract_features(window, sr)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # 🚨 REVERSE QUANTIZATION FOR OUTPUTS 🚨
            output_dtype = output_details[0]['dtype']
            if output_dtype in [np.int8, np.uint8]:
                scale, zero_point = output_details[0]['quantization']
                if scale > 0.0:
                    output_data = (output_data.astype(np.float32) - zero_point) * scale

            confidence_dict = {LABELS[i]: float(output_data[i]) for i in range(len(LABELS))}
            predicted_class = max(confidence_dict, key=confidence_dict.get)
            score = confidence_dict[predicted_class]

            for label in LABELS:
                best_confidence[label] = max(best_confidence[label], confidence_dict[label])

            if score > 0.85:
                if predicted_class != last_detected_class and cooldown == 0:
                    counts[predicted_class] += 1
                    last_detected_class = predicted_class
                    cooldown = 2

            if cooldown > 0:
                cooldown -= 1

        valid_counts = {k: v for k, v in counts.items() if v > 0}
        main_prediction = max(valid_counts, key=valid_counts.get) if valid_counts else "No clear sound detected"

        return {
            "prediction": main_prediction,
            "confidence": best_confidence,
            "counts": counts
        }

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        y = y * 32768.0
       
        all_windows = []
        for start in range(0, len(y), 8000):
            window = y[start:start + 16000]
            if len(window) < 1600:
                continue
                
            if np.max(np.abs(window)) < 250:
                all_windows.append({"window_start_sec": round(start/sr, 2), "skipped": "silence"})
                continue
                
            input_tensor = extract_features(window, sr)
            
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # 🚨 REVERSE QUANTIZATION FOR OUTPUTS 🚨
            output_dtype = output_details[0]['dtype']
            if output_dtype in [np.int8, np.uint8]:
                scale, zero_point = output_details[0]['quantization']
                if scale > 0.0:
                    output_data = (output_data.astype(np.float32) - zero_point) * scale
            
            scores = {LABELS[i]: round(float(output_data[i]), 4) for i in range(len(LABELS))}
            all_windows.append({"window_start_sec": round(start/sr, 2), "scores": scores})

        return {"windows": all_windows}
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)