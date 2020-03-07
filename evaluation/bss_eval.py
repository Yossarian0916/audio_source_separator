import os
import json
import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from utils import module_path
from utils.helper import istft
from utils.dataset import create_samples


# def get_separated_tracks(spectrogram_separator, phase_separator, mix_audio, offset=0.0, duration=None):
#     # load mix music audio, average the stereo recording to single channel audio track
#     # convert to spectrogram
#     sound, sr = librosa.load(mix_audio, sr=44100, mono=True, offset=offset, duration=duration)
#     stft = librosa.stft(sound, n_fft=2048, hop_length=512, win_length=2048)
#     mag = np.abs(stft)
#     phase_angle = np.angle(stft)
#     # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
#     mag_clips = np.empty((0, 1025, 100))
#     for i in range(mag.shape[1] // 100):
#         mag_clips = np.concatenate((mag_clips, mag[np.newaxis, :, i * 100: (i + 1) * 100]))
#     # chop phase angle of spectrogram into clips, each has 1025 bins, 100 frames
#     phase_clips = np.empty((0, 1025, 100))
#     for i in range(phase_angle.shape[1] // 100):
#         phase_clips = np.concatenate((phase_clips, phase_angle[np.newaxis, :, i * 100: (i + 1) * 100]))
#     # generated components spectrogram magnitude and phase_angle clips
#     separated_sepctrograms = spectrogram_separator.predict(mag_clips)
#     separated_phases = phase_separator.predict(phase_clips)
#     # separated_spectrograms contains 4 stem tracks
#     # the index of spectrograms: 0, 1, 2, 3 -> vocals, bass, drums, other
#     separated_tracks = list()
#     for i in range(4):
#         separated_spectrogram_magnitude = np.squeeze(separated_sepctrograms[i], axis=-1)
#         separated_spectrogram_phase_angle = np.squeeze(separated_phases[i], axis=-1)
#         track_waveform = istft(separated_spectrogram_magnitude, separated_spectrogram_phase_angle)
#         separated_tracks.append(track_waveform)
#     return separated_tracks


def get_separated_tracks(separator, mix_audio, offset=0.0, duration=None):
    # load mix music audio, average the stereo recording to single channel audio track
    # convert to spectrogram
    sound, sr = librosa.load(mix_audio, sr=44100, mono=True, offset=offset, duration=duration)
    stft = librosa.stft(sound, n_fft=2048, hop_length=512, win_length=2048)
    mag = np.abs(stft)
    phase = np.angle(stft)
    # chop magnitude of spectrogram into clips, each has 1025 bins, 100 frames
    stft_clips = np.empty((0, 1025, 100))
    for i in range(mag.shape[1] // 100):
        stft_clips = np.concatenate((stft_clips, mag[np.newaxis, :, i * 100: (i + 1) * 100]))
    # separate components from the mix single channel music audio
    separated_sepctrograms = separator.predict(stft_clips)
    # separated_spectrograms contains 4 stem tracks
    # the index of spectrograms: 0, 1, 2, 3 -> vocals, bass, drums, other
    separated_tracks = list()
    for i in range(4):
        separated_spectrogram = np.squeeze(separated_sepctrograms[i], axis=-1)
        spectrogram = np.concatenate(separated_spectrogram, axis=1)
        phase = phase[tuple(map(slice, spectrogram.shape))]
        reconstructed_track = librosa.istft(spectrogram * np.exp(1j * phase), hop_length=512, win_length=2048)
        separated_tracks.append(reconstructed_track)
    return separated_tracks


def get_reference_tracks(sample, track_shape, offset=0.0, duration=None):
    reference_tracks = list()
    feat_name = ['vocals', 'bass', 'drums', 'other']
    for feat in feat_name:
        track, sr = librosa.load(sample[feat], sr=44100, mono=True, offset=offset, duration=duration)
        # crop reference track to match separated track shape
        track = track[tuple(map(slice, track_shape))]
        reference_tracks.append(track)
    return reference_tracks


def get_normalization_baseline(sample, track_shape, offset=0.0, duration=None):
    """return a list of ['mix', 'mix', 'mix', 'mix'] as normalization base"""
    track, sr = librosa.load(sample['mix'], sr=44100, mono=True, offset=offset, duration=duration)
    # crop reference track to match separated track shape
    track = track[tuple(map(slice, track_shape))]
    mix_tracks = [track] * 4
    return mix_tracks


def estimate_and_evaluate(sample, separator_model, offset, duration):
    # separation tracks
    separated_tracks = get_separated_tracks(separator_model, sample['mix'], offset, duration)
    estimates = np.asarray(separated_tracks)
    track_length = separated_tracks[0].shape
    reference_tracks = get_reference_tracks(sample, track_length, offset, duration)
    for i in range(4):
        assert reference_tracks[i].shape == separated_tracks[i].shape
    references = np.asarray(reference_tracks)
    # run bss-eval
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
        references, estimates, compute_permutation=False)
    # get bss_eval metrics
    # between ['vocals', 'bass', 'drums', 'other'] and mixture ['mix', 'mix', 'mix', 'mix]
    # these metrics act as normalization baselines
    mix_tracks = get_normalization_baseline(sample, track_length, offset, duration)
    mix_tracks = np.asarray(mix_tracks)
    (baseline_sdr, baseline_sir, baseline_sar, _) = mir_eval.separation.bss_eval_sources(
        references, mix_tracks, compute_permutation=False)
    results = {
        'name': sample['name'],
        'sdr': sdr.tolist(),
        'sir': sir.tolist(),
        'sar': sar.tolist(),
        'nsdr': (sdr - baseline_sdr).tolist(),
    }
    return results


def write_results_to_json(sample, separator, model_name):
    results = estimate_and_evaluate(sample, separator, offset=30.0, duration=30.0)
    print(results)
    # save evaluation results to .json file
    save_file_name = '_'.join(['eval', sample['name']]) + '.json'
    evaluation_path = module_path.get_evaluation_path()
    save_results_path = os.path.join(evaluation_path, model_name)
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    save_results_json = os.path.join(save_results_path, save_file_name)
    with open(save_results_json, 'w', encoding='utf-8') as fd:
        json.dump(results, fd, sort_keys=True, indent=4)


def main(pre_trained_model_path):
    if pre_trained_model_path == '':
        raise AttributeError('the model path is empty!')
    else:
        _, model_name = os.path.split(pre_trained_model_path)
    # load pre-trained model
    saved_model_path = module_path.get_saved_model_path()
    model_path = os.path.join(saved_model_path, pre_trained_model_path)
    try:
        separator = tf.keras.models.load_model(model_path)
    except OSError:
        print('Please the saved model file name!')
    # load the whole dsd100 dataset
    train_samples = create_samples('Dev')
    test_samples = create_samples('Test')
    eval_samples = test_samples + train_samples
    # computing metrics
    print('\nGenerating evaluation metrics on dsd100 samples...')
    for i in range(len(eval_samples)):
        try:
            write_results_to_json(eval_samples[i], separator, model_name=model_name)
        except (ValueError):
            print(eval_samples[i]['name'] + ' is skipped')
            continue


if __name__ == '__main__':
    # main('conv_denoising_unet?time=20200307_1423.h5')
    main('conv_encoder_denoising_decoder?time=20200227_0838_l2_weight_regularization.h5')
