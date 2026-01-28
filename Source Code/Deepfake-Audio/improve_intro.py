import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

# Fix for "tf.placeholder() is not compatible with eager execution"
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

PROJ_DIR = Path(__file__).parent.absolute()
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

try:
    import encoder.inference
    import encoder.audio
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
except ImportError as e:
    print(f"Import failure: {e}")
    sys.exit(1)

# Models
ENC_MODEL = PROJ_DIR / "encoder/saved_models/pretrained.pt"
SYN_MODEL = PROJ_DIR / "synthesizer/saved_models/logs-pretrained/taco_pretrained"
VOC_MODEL = PROJ_DIR / "vocoder/saved_models/pretrained/pretrained.pt"

def main():
    print("Loading models...")
    encoder.inference.load_model(ENC_MODEL)
    synthesizer = Synthesizer(SYN_MODEL)
    vocoder.load_model(VOC_MODEL)
    
    ref_path = PROJ_DIR / "obama_enhanced.wav"
    if not ref_path.exists():
        print(f"Error: {ref_path} not found.")
        return

    print(f"Processing reference: {ref_path}")
    # Load first 10 seconds of the speech for optimal embedding
    original_wav, sampling_rate = librosa.load(ref_path, sr=None, duration=10)
    preprocessed_wav = encoder.audio.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.inference.embed_utterance(preprocessed_wav)
    
    text = "Hello. I'm Barack Obama. Welcome to the Deepfake Audio project. We live in an era where technology is changing the way we interact with the world. This studio, developed by Amey Thakur and Mega Satish, represents the intersection of innovation and creativity. Let's explore the future of neural synthesis together. Yes, we can."
    
    print("Synthesizing...")
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    
    print("Vocoding...")
    generated_wav = vocoder.infer_waveform(specs[0])
    
    # Add a bit of silence at the end
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    
    output_path = PROJ_DIR / "intro_message.wav"
    sf.write(output_path, generated_wav, synthesizer.sample_rate)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    main()
