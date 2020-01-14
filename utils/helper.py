import librosa
import numpy as np

import glob
import os
import sys
import math

from utils.config import STFT_CONFIG


# parameters
FREQ_BINS = STFT_CONFIG['FREQ_BINS']
TIME_FRAMES = STFT_CONFIG['TIME_FRAMES']
N_FFT = STFT_CONFIG['N_FFT']
HOP_LEN = STFT_CONFIG['HOP_LEN']
WIN_LEN = STFT_CONFIG['WIN_LEN']
SR = STFT_CONFIG['SR']
CLIP_LEN = STFT_CONFIG['CLIP_LEN']


def get_filenames(path):
    current_path = os.path.abspath(__file__)
    utils_path = os.path.dirname(current_path)
    root = os.path.dirname(utils_path)
    data_dir = os.path.join(root, 'data')
    file_path = os.path.join(data_dir, path)
    filenames = glob.glob(os.path.join(file_path), recursive=True)
    return filenames


def wav2stft(wav_file):
    """return absolute magnitude of STFT spectrum"""
    stft_clip = np.empty((0, FREQ_BINS, TIME_FRAMES))
    duration = librosa.get_duration(filename=wav_file)
    for i in range(math.floor(duration/CLIP_LEN)):
        sound, sr = librosa.load(
            wav_file, sr=SR, offset=i*CLIP_LEN, duration=CLIP_LEN)
        stft = librosa.stft(sound, n_fft=N_FFT,
                            hop_length=HOP_LEN, win_length=WIN_LEN)
        mag, stft = librosa.magphase(stft)
        stft_clip = np.concatenate((stft_clip, mag[np.newaxis, ...]), axis=0)
    return stft_clip


def wav2phase(wav_file):
    """return phase of STFT spectrum"""
    stft_phase = np.empty((0, FREQ_BINS, TIME_FRAMES))
    duration = librosa.get_duration(filename=wav_file)
    for i in range(math.floor(duration/CLIP_LEN)):
        sound, sr = librosa.load(
            wav_file, sr=SR, offset=i*CLIP_LEN, duration=CLIP_LEN)
        stft = librosa.stft(sound, n_fft=N_FFT,
                            hop_length=HOP_LEN, win_length=WIN_LEN)
        mag, phase = librosa.magphase(stft)
        stft_phase = np.concatenate(
            (stft_phase, phase[np.newaxis, ...]), axis=0)
    return stft_phase


def get_stft(path):
    """return np.ndarray containing STFT magnitude of wav files found in given path"""
    files = get_filenames(path)
    clips = np.empty((0, FREQ_BINS, TIME_FRAMES))
    for wav in files:
        stft_clip = wav2stft(wav)
        clips = np.concatenate((clips, stft_clip), axis=0)
    return clips


def get_phase(path):
    """return np.ndarray containing STFT phase info of wav files found in given path"""
    files = get_filenames(path)
    clips = np.empty((0, FREQ_BINS, TIME_FRAMES))
    for wav in files:
        stft_phase = wav2phase(wav)
        clips = np.concatenate((clips, stft_phase), axis=0)
    return clips
