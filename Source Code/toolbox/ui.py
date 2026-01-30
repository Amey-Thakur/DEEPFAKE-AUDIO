# ==================================================================================================
# DEEPFAKE AUDIO - toolbox/ui.py (Graphical Interface Framework)
# ==================================================================================================
# 
# 📝 DESCRIPTION
# This module implements the PyQt5-based Graphical User Interface for the 
# cloning toolbox. It handles the rendering of Mel-Spectrograms, UMAP 
# projections, audio device management, and the overall layout for an 
# interactive Neural Cloning experience.
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
from pathlib import Path
from time import sleep
from typing import List, Set
from warnings import filterwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import umap
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance

filterwarnings("ignore")

# UI Palette: Distinctive colors for speaker clustering
colormap = np.array([
    [0, 127, 70], [255, 0, 0], [255, 217, 38], [0, 135, 255], [165, 0, 165],
    [255, 167, 255], [97, 142, 151], [0, 255, 255], [255, 96, 38], [142, 76, 0],
    [33, 0, 127], [0, 0, 0], [183, 183, 183], [76, 255, 0],
], dtype=np.float64) / 255

default_text = \
    "Welcome to the toolbox! To begin, load an utterance from your datasets or record one " \
    "yourself.\nOnce its embedding has been created, you can synthesize any text written here.\n" \
    "The synthesizer expects to generate " \
    "outputs that are somewhere between 5 and 12 seconds.\nTo mark breaks, write a new line. " \
    "Each line will be treated separately.\nThen, they are joined together to make the final " \
    "spectrogram. Use the vocoder to generate audio.\nThe vocoder generates almost in constant " \
    "time, so it will be more time efficient for longer inputs like this one.\nOn the left you " \
    "have the embedding projections. Load or record more utterances to see them.\nIf you have " \
    "at least 2 or 3 utterances from a same speaker, a cluster should form.\nSynthesized " \
    "utterances are of the same color as the speaker whose voice was used, but they're " \
    "represented with a cross."

class UI(QDialog):
    """
    Interface Controller:
    Manages widget states, plotting routines, and audio playback. 
    Implements the visual representation of the neural cloning pipeline.
    """
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20

    def draw_utterance(self, utterance: Utterance, which):
        """View Synchronizer: Updates both spectrogram and embedding plots for a sample."""
        self.draw_spec(utterance.spec, which)
        self.draw_embed(utterance.embed, utterance.name, which)

    def draw_embed(self, embed, name, which):
        """Heatmap Renderer: Visualizes the 256D speaker embedding as a heatmap."""
        embed_ax, _ = self.current_ax if which == "current" else self.gen_ax
        embed_ax.figure.suptitle("" if embed is None else name)

        # Buffer Sweep: Clear existing colorbars/images
        if len(embed_ax.images) > 0:
            embed_ax.images[0].colorbar.remove()
        embed_ax.clear()

        # Render New Embedding
        if embed is not None:
            plot_embedding_as_heatmap(embed, embed_ax)
            embed_ax.set_title("embedding")
        embed_ax.set_aspect("equal", "datalim")
        embed_ax.set_xticks([])
        embed_ax.set_yticks([])
        embed_ax.figure.canvas.draw()

    def draw_spec(self, spec, which):
        """Spectrographic Renderer: Displays Mel-Spectrogram frames in the UI."""
        _, spec_ax = self.current_ax if which == "current" else self.gen_ax
        spec_ax.clear()

        if spec is not None:
            spec_ax.imshow(spec, aspect="auto", interpolation="none")
            spec_ax.set_title("mel spectrogram")

        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.figure.canvas.draw()
        if which != "current":
            self.vocode_button.setDisabled(spec is None)

    def draw_umap_projections(self, utterances: Set[Utterance]):
        """Feature Space Visualizer: Projects high-dimensional embeddings into 2D via UMAP."""
        self.umap_ax.clear()
        speakers = np.unique([u.speaker_name for u in utterances])
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        embeds = [u.embed for u in utterances]

        # Conditional Guard: Ensure sufficient data for projection
        if len(utterances) < self.min_umap_points:
            self.umap_ax.text(.5, .5, "Add %d more points to\ngenerate the projections" %
                              (self.min_umap_points - len(utterances)),
                              horizontalalignment='center', fontsize=15)
            self.umap_ax.set_title("")
        else:
            if not self.umap_hot:
                self.log("Drawing UMAP projections for the first time, this will take a few seconds.")
                self.umap_hot = True

            # Dimensionality Reduction
            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            projections = reducer.fit_transform(embeds)

            speakers_done = set()
            for projection, utterance in zip(projections, utterances):
                color = colors[utterance.speaker_name]
                mark = "x" if "_gen_" in utterance.name else "o"
                label = None if utterance.speaker_name in speakers_done else utterance.speaker_name
                speakers_done.add(utterance.speaker_name)
                self.umap_ax.scatter(projection[0], projection[1], c=[color], marker=mark, label=label)
            self.umap_ax.legend(prop={'size': 10})

        self.umap_ax.set_aspect("equal", "datalim")
        self.umap_ax.set_xticks([])
        self.umap_ax.set_yticks([])
        self.umap_ax.figure.canvas.draw()

    def save_audio_file(self, wav, sample_rate):
        """Export Handler: Prompts user and saves audio to disk."""
        dialog = QFileDialog()
        dialog.setDefaultSuffix(".wav")
        fpath, _ = dialog.getSaveFileName(
            parent=self,
            caption="Select a path to save the audio file",
            filter="Audio Files (*.flac *.wav)"
        )
        if fpath:
            if Path(fpath).suffix == "": fpath += ".wav"
            sf.write(fpath, wav, sample_rate)

    def setup_audio_devices(self, sample_rate):
        """Hardware Discovery: Detects and filters usable audio I/O devices."""
        input_devices, output_devices = [], []
        for device in sd.query_devices():
            try:
                sd.check_input_settings(device=device["name"], samplerate=sample_rate)
                input_devices.append(device["name"])
            except: pass
            try:
                sd.check_output_settings(device=device["name"], samplerate=sample_rate)
                output_devices.append(device["name"])
            except Exception as e:
                if not device["name"] in input_devices:
                    warn("Unsupported output device %s for the sample rate: %d \nError: %s" % (device["name"], sample_rate, str(e)))

        self.audio_in_device = input_devices[0] if input_devices else None
        if not input_devices: self.log("No audio input device detected. Recording may not work.")

        if not output_devices:
            self.log("No supported output audio devices were found! Audio output may not work.")
            self.audio_out_devices_cb.addItems(["None"])
            self.audio_out_devices_cb.setDisabled(True)
        else:
            self.audio_out_devices_cb.clear()
            self.audio_out_devices_cb.addItems(output_devices)
            self.audio_out_devices_cb.currentTextChanged.connect(self.set_audio_device)
        self.set_audio_device()

    def set_audio_device(self):
        """Device Swapper: Sets the active sounddevice I/O targets."""
        output_device = self.audio_out_devices_cb.currentText()
        if output_device == "None": output_device = None
        sd.default.device = (self.audio_in_device, output_device)

    def play(self, wav, sample_rate):
        """Audio Streamer: Pipes waveform data to the selected output device."""
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")

    def stop(self):
        """Kill-switch: Halts all active audio streams."""
        sd.stop()

    def record_one(self, sample_rate, duration):
        """Microphone Ingest: Records a fixed-duration segment from the active input."""
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)
        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            return None

        # Simulation of Progress: Keep UI responsive during blocking record
        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()

        self.log("Done recording.")
        self.record_button.setText("Record")
        self.record_button.setDisabled(False)
        return wav.squeeze()

    @property
    def current_dataset_name(self): return self.dataset_box.currentText()
    @property
    def current_speaker_name(self): return self.speaker_box.currentText()
    @property
    def current_utterance_name(self): return self.utterance_box.currentText()

    def browse_file(self):
        """File Picker: Opens a native dialog for selecting external audio samples."""
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        return Path(fpath[0]) if fpath[0] != "" else ""

    @staticmethod
    def repopulate_box(box, items, random=False):
        """Dynamic Switcher: Clears and refills combo box selections."""
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)

    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int, random=True):
        """Linguistic Explorer: Populates the dataset/speaker/utterance hierarchy."""
        # Dataset Selection Level
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                self.log("Warning: recognized datasets not found in %s" % datasets_root)
                [b.setDisabled(True) for b in [self.random_utterance_button, self.random_speaker_button, 
                                              self.random_dataset_button, self.utterance_box, 
                                              self.speaker_box, self.dataset_box, self.browser_load_button, 
                                              self.auto_next_checkbox]]
                return
            self.repopulate_box(self.dataset_box, datasets, random)

        # Speaker Selection Level
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)

        # Utterance Selection Level
        if level <= 2:
            utterances_root = datasets_root.joinpath(self.current_dataset_name, self.current_speaker_name)
            utterances = []
            for ext in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % ext))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)

    def browser_select_next(self):
        """Auto-advance: Selects the subsequent utterance in the list."""
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self): return self.encoder_box.itemData(self.encoder_box.currentIndex())
    @property
    def current_synthesizer_fpath(self): return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())
    @property
    def current_vocoder_fpath(self): return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    def populate_models(self, models_dir: Path):
        """Backend Inventory: Refreshes the list of available model checkpoints."""
        encoder_fpaths = list(models_dir.glob("*/encoder.pt"))
        if not encoder_fpaths: raise Exception("No encoder models found in %s" % models_dir)
        self.repopulate_box(self.encoder_box, [(f.parent.name, f) for f in encoder_fpaths])

        synthesizer_fpaths = list(models_dir.glob("*/synthesizer.pt"))
        if not synthesizer_fpaths: raise Exception("No synthesizer models found in %s" % models_dir)
        self.repopulate_box(self.synthesizer_box, [(f.parent.name, f) for f in synthesizer_fpaths])

        vocoder_fpaths = list(models_dir.glob("*/vocoder.pt"))
        vocoder_items = [(f.parent.name, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

    @property
    def selected_utterance(self): return self.utterance_history.itemData(self.utterance_history.currentIndex())

    def register_utterance(self, utterance: Utterance):
        """Session Logger: Adds a processed sample to the interaction history."""
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)

        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        [b.setDisabled(False) for b in [self.play_button, self.generate_button, self.synthesize_button]]

    def log(self, line, mode="newline"):
        """UI Console: Appends or updates log messages in the interface."""
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines: del self.logs[0]
        elif mode == "append": self.logs[-1] += line
        elif mode == "overwrite": self.logs[-1] = line
        self.log_window.setText('\n'.join(self.logs))
        self.app.processEvents()

    def set_loading(self, value, maximum=1):
        """Progress Streamer: Updates the global progress bar state."""
        self.loading_bar.setValue(value * 100)
        self.loading_bar.setMaximum(maximum * 100)
        self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()

    def populate_gen_options(self, seed, trim_silences):
        """Control Synchronizer: Configures generation parameters based on session state."""
        if seed is not None:
            self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        if not trim_silences:
            self.trim_silences_checkbox.setChecked(False)
            self.trim_silences_checkbox.setDisabled(True)

    def update_seed_textbox(self):
        """UI Guard: Toggles seed input based on checkbox status."""
        self.seed_textbox.setEnabled(self.random_seed_checkbox.isChecked())

    def reset_interface(self):
        """Canvas Wipe: Clears all visualizations and resets buttons to default state."""
        self.draw_embed(None, None, "current")
        self.draw_embed(None, None, "generated")
        self.draw_spec(None, "current")
        self.draw_spec(None, "generated")
        self.draw_umap_projections(set())
        self.set_loading(0)
        [b.setDisabled(True) for b in [self.play_button, self.generate_button, self.synthesize_button, 
                                      self.vocode_button, self.replay_wav_button, self.export_wav_button]]
        [self.log("") for _ in range(self.max_log_lines)]

    def __init__(self):
        """Constructor: Initializes the PyQt application and constructs the widget tree."""
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("SV2TTS toolbox")

        # Root Layout Definition
        root_layout = QGridLayout()
        self.setLayout(root_layout)

        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 0, 1, 2)

        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2, 1, 2)

        self.projections_layout = QVBoxLayout()
        root_layout.addLayout(self.projections_layout, 1, 0, 1, 1)

        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 1, 3)

        # Projection Canvas Initialization
        fig, self.umap_ax = plt.subplots(figsize=(3, 3), facecolor="#F0F0F0")
        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.projections_layout.addWidget(FigureCanvas(fig))
        self.umap_hot = False
        self.clear_button = QPushButton("Clear")
        self.projections_layout.addWidget(self.clear_button)

        # Browser Widget Construction
        i = 0
        self.dataset_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Dataset</b>"), i, 0)
        browser_layout.addWidget(self.dataset_box, i + 1, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Speaker</b>"), i, 1)
        browser_layout.addWidget(self.speaker_box, i + 1, 1)
        self.utterance_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Utterance</b>"), i, 2)
        browser_layout.addWidget(self.utterance_box, i + 1, 2)
        self.browser_load_button = QPushButton("Load")
        browser_layout.addWidget(self.browser_load_button, i + 1, 3)
        i += 2

        self.random_dataset_button = QPushButton("Random")
        browser_layout.addWidget(self.random_dataset_button, i, 0)
        self.random_speaker_button = QPushButton("Random")
        browser_layout.addWidget(self.random_speaker_button, i, 1)
        self.random_utterance_button = QPushButton("Random")
        browser_layout.addWidget(self.random_utterance_button, i, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        browser_layout.addWidget(self.auto_next_checkbox, i, 3)
        i += 1

        browser_layout.addWidget(QLabel("<b>Use embedding from:</b>"), i, 0)
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, i, 1, 1, 3)
        i += 1

        self.browser_browse_button = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button, i, 0)
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, i, 1)
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, i, 2)
        self.stop_button = QPushButton("Stop")
        browser_layout.addWidget(self.stop_button, i, 3)
        i += 1

        # Backend Selector Construction
        self.encoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)

        self.audio_out_devices_cb=QComboBox()
        browser_layout.addWidget(QLabel("<b>Audio Output</b>"), i, 3)
        browser_layout.addWidget(self.audio_out_devices_cb, i + 1, 3)
        i += 2

        # Output Management Construction
        browser_layout.addWidget(QLabel("<b>Toolbox Output:</b>"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        browser_layout.addWidget(self.waves_cb, i, 1)
        self.replay_wav_button = QPushButton("Replay")
        browser_layout.addWidget(self.replay_wav_button, i, 2)
        self.export_wav_button = QPushButton("Export")
        browser_layout.addWidget(self.export_wav_button, i, 3)
        i += 1

        # Visualization Canvas Clusters
        vis_layout.addStretch()
        gridspec_kw = {"width_ratios": [1, 4]}
        fig, self.current_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        fig, self.gen_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        for ax in self.current_ax.tolist() + self.gen_ax.tolist():
            ax.set_facecolor("#F0F0F0")
            [ax.spines[side].set_visible(False) for side in ["top", "right", "bottom", "left"]]

        # Text Prompt and Generation Controls
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)

        self.generate_button = QPushButton("Synthesize and vocode")
        gen_layout.addWidget(self.generate_button)

        h_layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        h_layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only")
        h_layout.addWidget(self.vocode_button)
        gen_layout.addLayout(h_layout)

        # Seed and Status Widgets
        layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed:")
        layout_seed.addWidget(self.random_seed_checkbox, 0, 0)
        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(80)
        layout_seed.addWidget(self.seed_textbox, 0, 1)
        self.trim_silences_checkbox = QCheckBox("Enhance vocoder output")
        layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        gen_layout.addLayout(layout_seed)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)

        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()

        # Window Geometry Finalization
        self.resize(self.app.primaryScreen().availableGeometry().size() * 0.8)
        self.reset_interface()
        self.show()

    def start(self):
        """Event Loop Launcher: Starts the PyQt application loop."""
        self.app.exec_()
