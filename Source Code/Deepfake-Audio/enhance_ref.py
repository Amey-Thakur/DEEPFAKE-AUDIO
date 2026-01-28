import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path

def enhance_audio(input_path, output_path):
    print(f"Analyzing {input_path}...")
    y, sr = librosa.load(input_path, sr=None)
    print(f"Original Sample Rate: {sr}")
    print(f"Length: {len(y)/sr:.2f} seconds")
    
    # 1. Trim silence
    y_trimmed, _ = librosa.effects.trim(y)
    
    # 2. Normalize
    y_norm = librosa.util.normalize(y_trimmed)
    
    # 3. Resample to 16kHz (standard for Tacotron 2)
    y_16k = librosa.resample(y_norm, orig_sr=sr, target_sr=16000)
    
    # 4. Save enhanced version
    sf.write(output_path, y_16k, 16000)
    print(f"Enhanced audio saved to {output_path}")

if __name__ == "__main__":
    ref_path = "obama_ref.wav"
    enhanced_path = "obama_enhanced.wav"
    if os.path.exists(ref_path):
        enhance_audio(ref_path, enhanced_path)
    else:
        print("Reference file not found.")
