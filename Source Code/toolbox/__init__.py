# ==================================================================================================
# DEEPFAKE AUDIO - toolbox/__init__.py (Interactive Logic Controller)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the main controller for the SV2TTS toolbox. It 
# orchestrates the interaction between the User Interface (PyQt5), the neural 
# encoder, synthesizer, and vocoder modules. It manages dataset browsing, 
# real-time voice recording, and the cloning workflow.
#
# 👤 AUTHORS
# - Amey Thakur (https://github.com/Amey-Thakur)
# - Mega Satish (https://github.com/msatmod)
#
# 🤝🏻 CREDITS
# Original Real-Time Voice Cloning methodology by CorentinJ
# Repository: https://github.com/CorentinJ/Real-Time-Voice-Cloning
#
# 🔗 PROJECT LINKS
# Repository: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO
# Video Demo: https://youtu.be/i3wnBcbHDbs
# Research: https://github.com/Amey-Thakur/DEEPFAKE-AUDIO/blob/main/DEEPFAKE-AUDIO.ipynb
#
# 📜 LICENSE
# Released under the MIT License
# Release Date: 2021-02-06
# ==================================================================================================

import sys
import traceback
from pathlib import Path
from time import perf_counter as timer

import numpy as np
import torch

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from toolbox.ui import UI
from toolbox.utterance import Utterance
from vocoder import inference as vocoder

# Dataset Catalog: Predefined paths for recognized speech datasets
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

# Memory Limit: Maximum cached waveforms to prevent OOM errors
MAX_WAVS = 15

class Toolbox:
    """
    Main Orchestrator:
    Binds neural inference backends to the graphical interface. 
    Handles application state, hardware interfacing, and error logging.
    """
    def __init__(self, datasets_root: Path, models_dir: Path, seed: int=None):
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav

        self.synthesizer = None # type: Synthesizer
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Feature Detection: Optional WebRTC Voice Activity Detection (VAD)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        self.ui = UI()
        self.reset_ui(models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        """Diagnostic Supervisor: Intercepts and logs application crashes to the UI."""
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        """Event Binder: Connects UI signals to toolbox control logic."""
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection and initialization
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        self.ui.synthesizer_box.currentIndexChanged.connect(lambda: setattr(self, 'synthesizer', None))
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)

        # File exploration and playback controls
        self.ui.browser_browse_button.clicked.connect(lambda: self.load_from_browser(self.ui.browse_file()))
        self.ui.utterance_history.currentIndexChanged.connect(lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current"))
        self.ui.play_button.clicked.connect(lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate))
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        # Audio and Persistence
        self.ui.setup_audio_devices(Synthesizer.sample_rate)
        self.ui.replay_wav_button.clicked.connect(self.replay_last_wav)
        self.ui.export_wav_button.clicked.connect(self.export_current_wave)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Synthesis and Parameterization
        self.ui.generate_button.clicked.connect(lambda: self.synthesize() or self.vocode())
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        """Archives the currently active waveform."""
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        """Persistence Bridge: Saves the currently active waveform to the filesystem."""
        self.ui.save_audio_file(self.current_wav, Synthesizer.sample_rate)

    def replay_last_wav(self):
        """Audio Feedback: Replays the generated cloning result."""
        self.ui.play(self.current_wav, Synthesizer.sample_rate)

    def reset_ui(self, models_dir: Path, seed: int=None):
        """Structural Sync: Refreshes models and dataset browser state."""
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(models_dir)
        self.ui.populate_gen_options(seed, self.trim_silences)

    def load_from_browser(self, fpath=None):
        """Asset Loader: Imports audio files from the dataset browser or local paths."""
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "": return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)
        self.add_real_utterance(wav, name, speaker_name)

    def record(self):
        """Input Bridge: Captures 5 seconds of live audio from the microphone."""
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None: return
        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)

    def add_real_utterance(self, wav, name, speaker_name):
        """Analytic Registration: Computes embedding and spectrogram for a real sample."""
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        if not encoder.is_loaded(): self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)

    def clear_utterances(self):
        """Memory Cleanse: Resets the projection view and session history."""
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self):
        """Neural Synthesis Phase: Generates Mel-Spectrogram frames from text."""
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)

        seed = int(self.ui.seed_textbox.text()) if self.ui.random_seed_checkbox.isChecked() else None
        if seed is not None:
            self.ui.populate_gen_options(seed, self.trim_silences)
            torch.manual_seed(seed)

        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        texts = self.ui.text_prompt.toPlainText().split("\n")
        embed = self.ui.selected_utterance.embed
        specs = self.synthesizer.synthesize_spectrograms(texts, [embed] * len(texts))
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        """Neural Vocoding Phase: Transforms Mel-Spectrograms into audible waveforms."""
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        seed = int(self.ui.seed_textbox.text()) if self.ui.random_seed_checkbox.isChecked() else None
        if seed is not None:
            self.ui.populate_gen_options(seed, self.trim_silences)
            torch.manual_seed(seed)

        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)

        if self.ui.current_vocoder_fpath is not None:
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
            
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")

        # Concat with breath breaks and post-process
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breath_breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breath_breaks) for i in (w, b)])

        if self.ui.trim_silences_checkbox.isChecked():
            wav = encoder.preprocess_wav(wav)

        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)

        # Update Session History
        wav_name = str(self.waves_count + 1)
        self.waves_count += 1
        if self.waves_count > MAX_WAVS:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)
        self.set_current_wav(0)

        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Clone Verification: Compute embedding for the generated voice
        if not encoder.is_loaded(): self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)

    def init_encoder(self):
        """Neural Wake-up: Loads the speaker encoder model into memory."""
        model_fpath = self.ui.current_encoder_fpath
        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_synthesizer(self):
        """Neural Wake-up: Loads the Tacotron-2 synthesizer model."""
        model_fpath = self.ui.current_synthesizer_fpath
        self.ui.log("Loading the synthesizer %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        self.synthesizer = Synthesizer(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        """Neural Wake-up: Loads the WaveRNN vocoder model."""
        model_fpath = self.ui.current_vocoder_fpath
        if model_fpath is None: return
        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def update_seed_textbox(self):
        """UI Logic: Synchronizes seed control state."""
        self.ui.update_seed_textbox()
