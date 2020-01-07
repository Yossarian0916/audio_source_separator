import librosa
import numpy as np
import tensorflow as tf

import os
import sys

from utils.helper import get_stft, istft, get_filenames


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
test_song_stft = get_stft(test_song)
test_song_stft = tf.expand_dims(test_song_stft, axis=1)
print(test_song_stft.shape)

# load saved model
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
saved_model_dir = os.path.join(root, "notebook", "encoder_decoder_separator")
saved_model_path = os.path.join(
    saved_model_dir, "encoder_decoder_separator_2020-01-07_08:03")
separator = tf.saved_model.load(saved_model_path)
output = [separator.call(song_stft_clip) for song_stft_clip in test_song_stft]
