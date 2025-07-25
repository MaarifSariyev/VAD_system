import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('vad_cnn_model.h5')

# Audio parameters
sr = 16000          # Sampling rate
chunk_duration = 1  # seconds per chunk
chunk_size = sr * chunk_duration

# MFCC parameters
n_mfcc = 13
max_len = 200  # your model input length

def preprocess_audio(audio_chunk):
    # Convert to mono if stereo
    if len(audio_chunk.shape) > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)
    # Resample if needed (assuming audio_chunk is already sr)
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=n_mfcc).T
    # Pad or truncate
    if len(mfcc) > max_len:
        mfcc = mfcc[:max_len]
    else:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0,0)), mode='constant')
    return mfcc[np.newaxis, ...]

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_data = indata[:, 0]  # Take first channel if stereo
    mfcc_input = preprocess_audio(audio_data)
    prob = model.predict(mfcc_input)[0][0]
    is_speech = prob > 0.5
    print(f"Speech Detected: {is_speech} (Prob: {prob:.3f})")

# Start streaming from microphone
with sd.InputStream(channels=1, samplerate=sr, blocksize=chunk_size, callback=audio_callback):
    print("Listening (press Ctrl+C to stop)...")
    import time
    while True:
        time.sleep(0.1)
