
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import gradio as gr
from encoder import inference as encoder_inference
from encoder import audio as encoder_audio  # Correct import for preprocessing
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# --- Configuration & Paths ---
PROJ_DIR = Path(__file__).parent
ENC_MODEL = PROJ_DIR / "encoder/saved_models/pretrained.pt"
SYN_MODEL = PROJ_DIR / "synthesizer/saved_models/logs-pretrained/taco_pretrained"
VOC_MODEL = PROJ_DIR / "vocoder/saved_models/pretrained/pretrained.pt"

# --- Model Loading ---
print("Initializing AI Models...")
try:
    encoder_inference.load_model(ENC_MODEL)
    synthesizer = Synthesizer(SYN_MODEL)
    vocoder.load_model(VOC_MODEL)
    print("Models Loaded Successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # We continue so the UI launches, but inference will fail if models are missing

# --- Inference Pipeline ---
def clone_voice(audio_file, text, progress=gr.Progress()):
    if audio_file is None:
        return None, "⚠️ Please upload a reference voice sample."
    if not text.strip():
        return None, "⚠️ Please enter text to synthesize."

    try:
        progress(0.1, desc="Preprocessing Audio")
        # 1. Preprocess the reference audio
        original_wav, sampling_rate = sf.read(audio_file)
        # Use the explicit audio module for preprocessing
        preprocessed_wav = encoder_audio.preprocess_wav(original_wav, sampling_rate)
        
        progress(0.3, desc="Generating Speaker Embedding")
        # 2. Embed the utterance (Encoder)
        embed = encoder_inference.embed_utterance(preprocessed_wav)
        
        progress(0.5, desc="Synthesizing Mel Spectrogram")
        # 3. Synthesize the spectrogram (Synthesizer)
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        spec = specs[0]
        
        progress(0.8, desc="Vocoding Waveform")
        # 4. Generate waveform (Vocoder)
        generated_wav = vocoder.infer_waveform(spec)
        
        # Pad silence
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        
        progress(1.0, desc="Complete")
        return (synthesizer.sample_rate, generated_wav), "✅ Synthesis Successful"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# --- Custom CSS & Theme ---
# Scholarly aesthetics: Clean, high contrast text, subtle shadows, professional spacing
custom_css = """
body { background-color: #f4f6f9; }
.container { max-width: 1200px; margin: auto; padding-top: 20px; }
#header { text-align: center; margin-bottom: 30px; }
#header h1 { font-family: 'Georgia', serif; font-size: 2.5em; color: #2c3e50; margin-bottom: 10px; }
#header p { font-family: 'Helvetica Neue', sans-serif; font-size: 1.1em; color: #7f8c8d; }
.gr-box { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e1e4e8; background: white; }
.gr-button { border-radius: 6px; transition: all 0.3s ease; }
.gr-button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
#status-box { font-family: monospace; }
"""

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="stone",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#2c3e50",
    button_primary_background_fill_hover="#34495e",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 2px 4px rgba(0,0,0,0.05)"
)

# --- UI Layout (Blocks) ---
with gr.Blocks(title="Deepfake Audio Research") as demo:
    with gr.Column(elem_classes=["container"]):
        
        # Header
        with gr.Column(elem_id="header"):
            gr.Markdown("""
            # Deepfake Audio Research
            ### Real-Time Voice Cloning & Text-to-Speech Synthesis
            """)
            gr.Markdown("An implementation of Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS).")

        # Main Interface
        with gr.Row(equal_height=True):
            
            # Left Column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference Audio")
                audio_input = gr.Audio(
                    type="filepath", 
                    label="Upload Reference Voice", 
                    elem_classes=["gr-box"]
                )
                
                gr.Markdown("### 2. Input Text")
                text_input = gr.Textbox(
                    label="Text Content", 
                    placeholder="Enter the text you wish to synthesize...", 
                    lines=5, 
                    elem_classes=["gr-box"]
                )
                
                generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

            # Right Column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 3. Generated Result")
                audio_output = gr.Audio(
                    label="Synthesized Speech", 
                    interactive=False, 
                    elem_classes=["gr-box"]
                )
                
                gr.Markdown("### System Status")
                status_output = gr.Label(
                    value="Ready", 
                    label="Execution Log", 
                    elem_classes=["gr-box", "status-box"]
                )

        # Footer
        with gr.Row():
            gr.Markdown("""
            ---
            **Project Authors**: Amey Thakur & Mega Satish | **License**: MIT  
            *Based on SV2TTS, Tacotron 2, and WaveRNN architectures.*
            """)

    # Event Binding
    generate_btn.click(
        fn=clone_voice,
        inputs=[audio_input, text_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.queue().launch(theme=theme, css=custom_css)
