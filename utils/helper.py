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


def istft(magnitude, phase=None, rebuild_iter=10, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN):
    if phase is not None:
        if rebuild_iter > 0:
            # refine audio given initial phase with a number of iterations
            return rebuild_phase(magnitude, n_fft, hop_length, win_length, rebuild_iter, phase)
        # reconstructing the new complex matrix
        stft_complx_matrix = magnitude * \
            np.exp(phase * 1j)  # magnitude * e^(j*phase)
        audio = librosa.istft(stft_complx_matrix, hop_length, win_length)
    else:
        audio = rebuild_phase(magnitude, n_fft, hop_length,
                              win_length, rebuild_iter)
    return audio


def rebuild_phase(magnitude, n_fft, hop_length, win_length, rebuild_iter=10, init_phase=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram,
    optionally with a given intial phase.
    '''
    for i in range(rebuild_iter):
        if i == 0:
            if init_phase is None:
                reconstruction = np.random.random_sample(
                    magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                # e^(j*phase), so that angle => phase
                reconstruction = np.exp(init_phase * 1j)
        else:
            reconstruction = librosa.stft(reconstruction, n_fft, hop_length)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == rebuild_iter - 1:
            audio = librosa.istft(spectrum, hop_length, win_length)
        else:
            audio = librosa.istft(spectrum, hop_length)
    return audio
