import os
import sys
import librosa
import museval
import numpy as np
import tensorflow as tf
import IPython.display as ipd
from utils.helper import wav_to_spectrogram_clips, rebuild_audio_from_spectro_clips
from utils.dataset import create_samples
from models.conv_denoising_unet import ConvDenoisingUnet
from models.conv_encoder_denoising_decoder import ConvEncoderDenoisingDecoder
from models.conv_resblock_denoising_unet import ConvResblockDenoisingUnet
from evaluation import evaluate
import mir_eval
from utils import module_path


samples = create_samples('Dev')
test_sample = samples[0]
print('test sample name: ', test_sample['name'])
saved_model_path = module_path.get_saved_model_path()
model_path = os.path.join(
    saved_model_path, 'conv_encoder_denoising_decoder?time=20200224_0618.h5')
model = tf.keras.models.load_model(model_path)


def get_separated_tracks(separator, mix_audio):
    # load mix music audio, average the stereo recording to single channel audio track
    # convert to spectrogram
    sound, sr = librosa.load(mix_audio, sr=44100, mono=True, duration=60)
    stft = librosa.stft(sound, n_fft=2048, hop_length=512, win_length=2048)
    mag, phase = librosa.magphase(stft)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    stft_clips = np.empty((0, 1025, 100))
    for i in range(mag.shape[1] // 100):
        stft_clips = np.concatenate((stft_clips, mag[np.newaxis, :, i * 100: (i + 1) * 100]))
    # separate components from the mix single channel music audio
    separated_sepctrograms = separator.predict(stft_clips)
    separated_tracks = list()
    # separated_spectrograms contains 4 stem tracks
    # the index of spectrograms: 0, 1, 2, 3 -> vocals, bass, drums, other
    for i in range(4):
        separated_track = np.squeeze(separated_sepctrograms[i], axis=-1)
        separated_tracks.append(rebuild_audio_from_spectro_clips(separated_track))
    return separated_tracks


def get_reference_tracks(sample, track_shape):
    reference_tracks = list()
    # feat_name = ['vocals', 'bass', 'drums', 'other']
    feat_name = ['mix'] * 4
    for feat in feat_name:
        track, sr = librosa.load(sample[feat], sr=44100, mono=True, duration=60)
        # crop reference track to match separated track shape
        track = track[tuple(map(slice, track_shape))]
        reference_tracks.append(track)
    return reference_tracks


separated_tracks = get_separated_tracks(model, test_sample['mix'])
reference_tracks = get_reference_tracks(test_sample, separated_tracks[0].shape)

results = mir_eval.separation.bss_eval_sources(np.asarray(
    reference_tracks), np.asarray(separated_tracks), compute_permutation=True)
print(results)
