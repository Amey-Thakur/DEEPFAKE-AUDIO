"""
Deepfake Audio - Plotting Utilities
-----------------------------------
Functions for plotting spectrograms and alignments.
Used for visualizing training progress and synthesis results.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def split_title_line(title_text, max_words=5):
	"""
	Splits the title text into multiple lines if it exceeds the max_words limit.
	Helps in keeping the plot titles readable.
	"""
	seq = title_text.split()
	return "\n".join([" ".join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
	"""
	Plots the attention alignment matrix.
	
	Args:
		alignment: Numpy array containing the alignment weights.
		path: Path (string) to save the plot.
		title: Title of the plot.
		split_title: Boolean, whether to split the title into multiple lines.
		max_len: Optional integer to truncate the alignment length (decoder steps).
	"""
	if max_len is not None:
		alignment = alignment[:, :max_len]

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111)

	im = ax.imshow(
		alignment,
		aspect="auto",
		origin="lower",
		interpolation="none")
	fig.colorbar(im, ax=ax)
	xlabel = "Decoder timestep"

	if split_title:
		title = split_title_line(title)

	plt.xlabel(xlabel)
	plt.title(title)
	plt.ylabel("Encoder timestep")
	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()


def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
	"""
	Plots the predicted mel-spectrogram and optionally the target spectrogram.
	
	Args:
		pred_spectrogram: Numpy array of the predicted spectrogram.
		path: Path (string) to save the plot.
		title: Title of the plot.
		split_title: Boolean, whether to split the title.
		target_spectrogram: Optional numpy array of the ground truth spectrogram.
		max_len: Optional integer to truncate the time axis.
		auto_aspect: Boolean, whether to use 'auto' aspect ratio for imshow.
	"""
	if max_len is not None:
		target_spectrogram = target_spectrogram[:max_len]
		pred_spectrogram = pred_spectrogram[:max_len]

	if split_title:
		title = split_title_line(title)

	fig = plt.figure(figsize=(10, 8))
	# Set common labels
	fig.text(0.5, 0.18, title, horizontalalignment="center", fontsize=16)

	#target spectrogram subplot
	if target_spectrogram is not None:
		ax1 = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)

		if auto_aspect:
			im = ax1.imshow(np.rot90(target_spectrogram), aspect="auto", interpolation="none")
		else:
			im = ax1.imshow(np.rot90(target_spectrogram), interpolation="none")
		ax1.set_title("Target Mel-Spectrogram")
		fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
		ax2.set_title("Predicted Mel-Spectrogram")
	else:
		ax2 = fig.add_subplot(211)

	if auto_aspect:
		im = ax2.imshow(np.rot90(pred_spectrogram), aspect="auto", interpolation="none")
	else:
		im = ax2.imshow(np.rot90(pred_spectrogram), interpolation="none")
	fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)

	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()
