import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load model
model = tf.keras.models.load_model('vad_cnn_model.h5')

# Audio parameters
sr = 16000
chunk_duration = 1  # seconds
chunk_size = sr * chunk_duration

# MFCC params
n_mfcc = 13
max_len = 200

# Buffer to store audio for plotting
audio_buffer = np.zeros(chunk_size * 10)  # 10 seconds buffer
time_axis = np.linspace(-10, 0, audio_buffer.size)

# VAD result buffer (one bool per chunk)
vad_buffer = []

# Prepare plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(time_axis, audio_buffer)
speech_patch = ax.axvspan(-chunk_duration, 0, color='green', alpha=0.3)
ax.set_ylim([-1, 1])
ax.set_xlabel("Time (s)")
ax.set_title("Live Audio Waveform with VAD (green=speech)")

def preprocess_audio(audio_chunk):
    if len(audio_chunk.shape) > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) > max_len:
        mfcc = mfcc[:max_len]
    else:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0,0)), mode='constant')
    return mfcc[np.newaxis, ...]

def update_plot(indata):
    global audio_buffer, vad_buffer, speech_patch

    # Shift audio buffer and add new chunk
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    # Predict speech
    mfcc_input = preprocess_audio(indata[:, 0])
    prob = model.predict(mfcc_input, verbose=0)[0][0]
    is_speech = prob > 0.5
    vad_buffer.append(is_speech)

    # Update waveform plot
    line.set_ydata(audio_buffer)

    # Update speech highlight span (show last chunk)
    if is_speech:
        speech_patch.xy = [(time_axis[-chunk_size], -1),
                           (time_axis[-chunk_size], 1),
                           (time_axis[-1], 1),
                           (time_axis[-chunk_size], -1)]
        speech_patch.set_visible(True)
    else:
        speech_patch.set_visible(False)

    fig.canvas.draw()
    fig.canvas.flush_events()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    update_plot(indata)

# Start audio stream and plot
with sd.InputStream(channels=1, samplerate=sr, blocksize=chunk_size, callback=audio_callback):
    print("Listening and visualizing VAD (press Ctrl+C to stop)...")
    try:
        while True:
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Stopped.")
