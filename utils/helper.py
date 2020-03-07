import librosa
import numpy as np
import glob
import os
import sys
import math


# parameters
N_FFT = 2048
HOP_LEN = 512
WIN_LEN = 2048
SR = 44100
FREQ_BINS = WIN_LEN // 2 + 1
TIME_FRAMES = 100


def wav_to_spectrogram_clips(wav_file):
    """convert audio into spectorgram, then chop it into 2d-segmentation of 100 frames"""
    # convert audio into spectorgram
    sound, sr = librosa.load(wav_file, sr=SR, mono=True)
    stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    mag, phase = librosa.magphase(stft)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    stft_clips = np.empty((0, FREQ_BINS, 100))
    for i in range(mag.shape[1] // 100):
        stft_clips = np.concatenate((stft_clips, mag[np.newaxis, :, i * 100: (i + 1) * 100]))
    return stft_clips


def wav_to_log_spectrogram_clips(wav_file):
    """convert audio into logrithmic spectorgram, then chop it into 2d-segmentation of 100 frames"""
    # convert audio into spectorgram
    sound, sr = librosa.load(wav_file, sr=SR, mono=True)
    stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    mag, phase = librosa.magphase(stft)
    db_spectro = librosa.amplitude_to_db(mag)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    db_spectro_clips = np.empty((0, FREQ_BINS, 100))
    for i in range(math.floor(mag.shape[1] / 100)):
        db_spectro_clips = np.concatenate((db_spectro_clips, db_spectro[np.newaxis, :, i * 100: (i + 1) * 100]))
    return db_spectro_clips


def rebuild_audio_from_spectro_clips(spectrogram_clips, is_dB_format=False):
    # audio spectrogram format:
    # 1. normal stft spectromgram
    # 2. dB-scaled spectrogram log(epilon + S*2)
    spectrogram = np.concatenate(spectrogram_clips, axis=1)
    if is_dB_format:
        spectrogram = librosa.db_to_amplitude(spectrogram)
    # reconstructed = librosa.griffinlim(spectrogram)
    reconstructed = librosa.istft(spectrogram, hop_length=512, win_length=2048)
    return reconstructed


def get_data_dir_filenames(path):
    current_path = os.path.abspath(__file__)
    utils_path = os.path.dirname(current_path)
    root = os.path.dirname(utils_path)
    data_dir = os.path.join(root, 'data')
    file_path = os.path.join(data_dir, path)
    filenames = glob.glob(os.path.join(file_path), recursive=True)
    return filenames
