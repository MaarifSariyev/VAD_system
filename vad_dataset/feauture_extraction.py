import os
import librosa
import numpy as np

# Paths
speech_dir = 'processed/speech'
non_speech_dir = 'processed/non_speech'
features_dir = 'features'

os.makedirs(features_dir, exist_ok=True)

# Parameters
sr = 16000  # sampling rate
n_mfcc = 13  # number of MFCC features

X = []  # feature list
y = []  # label list

def extract_features_from_dir(directory, label):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            audio, _ = librosa.load(filepath, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfcc = mfcc.T  # (frames, features)
            X.append(mfcc)
            y.append(label)

# Extract speech features
extract_features_from_dir(speech_dir, label=1)

# Extract non-speech features
extract_features_from_dir(non_speech_dir, label=0)

# Pad sequences to same length (or truncate)
max_len = 200  # adjust based on average length
X_padded = np.array([x[:max_len] if len(x) > max_len else np.pad(x, ((0, max_len - len(x)), (0, 0)), mode='constant') for x in X])
y_array = np.array(y)

# Save
np.save(os.path.join(features_dir, 'vad_features.npy'), X_padded)
np.save(os.path.join(features_dir, 'vad_labels.npy'), y_array)

print("✅ Feature extraction complete! Saved in vad_dataset/features/")
