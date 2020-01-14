import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def read_wav(file, duration):
    # read wav file, duration(in seconds)
    time_series, sample_rate = librosa.load(file, duration=duration)
    return time_series, sample_rate


def stft(time_series):
    stft_matrix = librosa.stft(time_series)
    magnitude, phase = librosa.magphase(stft_matrix)
    return magnitude, phase


def show_waves(sound, sample_rate):
    librosa.display.waveplot(np.array(sound), sample_rate)
    plt.show()


def show_stft_log_spec(sound, sample_rate):
    # compute stft spectrogram, y axis in logrithmic scale
    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(sound)), ref=np.max)
    librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
    # plot
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def show_melspecgram(sound, sample_rate):
    # compute mel-spectrogram, y axis is in logrithmic scale
    melspec = librosa.feature.melspectrogram(sound, sample_rate)
    power_spec_db = librosa.power_to_db(np.abs(melspec), ref=np.max)
    librosa.display.specshow(power_spec_db, x_axis='time', y_axis='mel')
    # plot
    plt.colorbar(format='%+2.0f dB')
    plt.show()
