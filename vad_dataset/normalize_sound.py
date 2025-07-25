import os
import librosa
import soundfile as sf

def normalize_and_trim(input_dir, output_dir, sr=16000, max_duration=30):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav') or f.endswith('.mp3')]
    for file in files:
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")
        try:
            y, _ = librosa.load(in_path, sr=sr, mono=True)
            y = y[:sr * max_duration]  # trim
            sf.write(out_path, y, sr)
            print(f"✅ Processed: {file}")
        except Exception as e:
            print(f"❌ Error: {file} — {e}")

# Azerbaijani Speech
normalize_and_trim("raw/az_speech", "processed/speech")

# MUSAN Noise + Music → all treated as non-speech
normalize_and_trim("raw/musan_noise", "processed/non_speech")
normalize_and_trim("raw/musan_music", "processed/non_speech")
