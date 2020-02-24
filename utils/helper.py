import librosa
import numpy as np
import glob
import os
import sys
import math
from utils.config import STFT_CONFIG


# parameters
N_FFT = STFT_CONFIG['N_FFT']
HOP_LEN = STFT_CONFIG['HOP_LEN']
WIN_LEN = STFT_CONFIG['WIN_LEN']
SR = STFT_CONFIG['SR']
CLIP_LEN = STFT_CONFIG['CLIP_LEN']

FREQ_BINS = WIN_LEN // 2 + 1
TIME_FRAMES = math.ceil(HOP_LEN / SR)


def wav_to_spectrogram_clips(wav_file):
    """convert audio into spectorgram, then chop it into 2d-segmentation of 100 frames"""
    # convert audio into spectorgram
    sound, sr = librosa.load(wav_file, sr=SR)
    stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    mag, phase = librosa.magphase(stft)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    stft_clips = np.empty((0, FREQ_BINS, 100))
    for i in range(math.floor(mag.shape[1] / 100)):
        stft_clips = np.concatenate((stft_clips, mag[np.newaxis, :, i * 100: (i + 1) * 100]))
    return stft_clips


def wav_to_log_spectrogram_clips(wav_file):
    """convert audio into logrithmic spectorgram, then chop it into 2d-segmentation of 100 frames"""
    # convert audio into spectorgram
    sound, sr = librosa.load(wav_file, sr=SR)
    stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    mag, phase = librosa.magphase(stft)
    db_spectro = librosa.amplitude_to_db(mag)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    db_spectro_clips = np.empty((0, FREQ_BINS, 100))
    for i in range(math.floor(mag.shape[1] / 100)):
        db_spectro_clips = np.concatenate((db_spectro_clips, db_spectro[np.newaxis, :, i * 100: (i + 1) * 100]))
    return db_spectro_clips


def wav_clips_to_spectrogram(wav_file):
    """
    chop audio file into 2-second long clip,
    then convert to waveform array,
    then convert to spectrogram
    """
    stft_clips = np.empty((0, FREQ_BINS, TIME_FRAMES))
    duration = librosa.get_duration(filename=wav_file)
    for i in range(math.floor(duration/CLIP_LEN)):
        sound, sr = librosa.load(wav_file, sr=SR, offset=i * CLIP_LEN, duration=CLIP_LEN)
        stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
        mag, phase = librosa.magphase(stft)
        stft_clips = np.concatenate((stft_clips, mag[np.newaxis, ...]), axis=0)
    return stft_clips


def wav_clips_to_log_spectrogram(wav_file):
    """
    chop audio file into 2-second long clip,
    then convert to waveform array,
    then convert to log(1+spectrogram) of a given audio
    """
    db_spectro_clips = np.empty((0, FREQ_BINS, TIME_FRAMES))
    duration = librosa.get_duration(filename=wav_file)
    for i in range(math.floor(duration / CLIP_LEN)):
        sound, sr = librosa.load(wav_file, sr=SR, offset=i * CLIP_LEN, duration=CLIP_LEN)
        stft = librosa.stft(sound, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
        mag, phase = librosa.magphase(stft)
        db_spectro = librosa.amplitude_to_db(mag)
        db_spectro_clips = np.concatenate((db_spectro_clips, db_spectro[np.newaxis, ...]), axis=0)
    return db_spectro_clips


def rebuild_audio_from_spectro_clips(audio_clips, phase_clips=None, is_dB_format=False):
    # audio spectrogram format:
    # 1. normal stft spectromgram
    # 2. dB-scaled spectrogram log(epilon + S*2)
    if is_dB_format:
        map_fn = librosa.db_to_amplitude
    else:
        def map_fn(x):
            return x
    # if no phase information, use griffin lim algorithm to rebuild phase from magnitude
    if phase_clips is None:
        reconstructed = [librosa.griffinlim(map_fn(clip)) for clip in audio_clips]
    else:
        reconstructed = [librosa.istft(clip*phase_clip) for (clip, phase_clip) in zip(audio_clips, phase_clips)]
    # string together all audio clips
    reconstructed = np.concatenate(reconstructed, axis=0)
    return reconstructed


def get_data_dir_filenames(path):
    current_path = os.path.abspath(__file__)
    utils_path = os.path.dirname(current_path)
    root = os.path.dirname(utils_path)
    data_dir = os.path.join(root, 'data')
    file_path = os.path.join(data_dir, path)
    filenames = glob.glob(os.path.join(file_path), recursive=True)
    return filenames
