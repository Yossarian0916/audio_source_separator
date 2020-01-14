import librosa
import numpy as np
import tensorflow as tf

import os
import sys

from utils.helper import get_stft, get_filenames
from evaluation.metrics import bss_eval


# loading data from disks
current_path = os.path.abspath(__file__)
evaluation_path = os.path.dirname(current_path)
root = os.path.dirname(evaluation_path)
data_dir = os.path.join(root, 'data')

dsd100_mix = os.path.join(data_dir, 'DSD100/Mixtures/Test')
dsd100_songs = [os.path.join(song_path, 'mixture.wav')
                for song_path in get_filenames(dsd100_mix+'/*')]

dsd100_stem = os.path.join(data_dir, 'DSD100/Sources/Test')
dsd100_bass = [os.path.join(song_path, 'bass.wav')
               for song_path in get_filenames(dsd100_stem+'/*')]
dsd100_drums = [os.path.join(song_path, 'drums.wav')
                for song_path in get_filenames(dsd100_stem+'/*')]
dsd100_other = [os.path.join(song_path, 'other.wav')
                for song_path in get_filenames(dsd100_stem+'/*')]
dsd100_vocals = [os.path.join(song_path, 'vocals.wav')
                 for song_path in get_filenames(dsd100_stem+'/*')]

# prepare one song to test separator quality
test_song = dsd100_songs[0]
test_song_vocal = dsd100_vocals[0]
test_song_bass = dsd100_bass[0]
test_song_drum = dsd100_drums[0]
test_song_other = dsd100_other[0]

test_song_stft = get_stft(test_song)
test_song_stft = tf.expand_dims(test_song_stft, axis=1)

# load saved model
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
saved_model_dir = os.path.join(root, "notebook", "checkpoints")
saved_model_path = os.path.join(
    saved_model_dir, "dae_reconstruction_separator", "dae_recon?optimizer=SGD?loss=MSE?lr=0.01?time=2020-01-09_08:44")
separator = tf.saved_model.load(saved_model_path)

separator_output = list()
for i in range(len(test_song_stft)):
    audio = tf.convert_to_tensor(
        test_song_stft[None, i, :, :], dtype=tf.float32)
    separator_output.append(separator.call(audio))
output_audio = np.concatenate(separator_output, axis=0)


def concat(audio_track):
    reconstructed = [librosa.griffinlim(clip) for clip in audio_track]
    reconstructed = np.concatenate(reconstructed, axis=0)
    return reconstructed


vocal_track = concat(output_audio[:, 0, :, :])
bass_track = concat(output_audio[:, 1, :, :])
drum_track = concat(output_audio[:, 2, :, :])
other_track = concat(output_audio[:, 3, :, :])

# evaluation, bss_eval
