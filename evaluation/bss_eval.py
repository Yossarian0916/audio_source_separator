import os
import json
import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from utils import module_path
from utils.helper import wav_to_spectrogram_clips, rebuild_audio_from_spectro_clips
from utils.dataset import create_samples


def get_separated_tracks(separator, mix_audio, offset=10, duration=90):
    # load mix music audio, average the stereo recording to single channel audio track
    # convert to spectrogram
    sound, sr = librosa.load(mix_audio, sr=44100, mono=True, offset=offset, duration=duration)
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


def get_reference_tracks(sample, track_shape, offset=10, duration=90):
    reference_tracks = list()
    feat_name = ['vocals', 'bass', 'drums', 'other']
    for feat in feat_name:
        track, sr = librosa.load(sample[feat], sr=44100, mono=True, offset=offset, duration=duration)
        # crop reference track to match separated track shape
        track = track[tuple(map(slice, track_shape))]
        reference_tracks.append(track)
    return reference_tracks


def get_normalization_baseline(sample, track_shape, offset=10, duration=90):
    """return a list of ['mix', 'mix', 'mix', 'mix'] as normalization base"""
    track, sr = librosa.load(sample['mix'], sr=44100, mono=True, offset=offset, duration=duration)
    # crop reference track to match separated track shape
    track = track[tuple(map(slice, track_shape))]
    baseline_tracks = [track] * 4
    return baseline_tracks


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
        references, estimates, compute_permutation=True)
    # get bss_eval metrics
    # between ['vocals', 'bass', 'drums', 'other'] and mixture ['mix', 'mix', 'mix', 'mix]
    # these metrics act as normalization baselines
    baselines = get_normalization_baseline(sample, track_length, offset, duration)
    baselines = np.asarray(baselines)
    (baseline_sdr, baseline_sir, baseline_sar, _) = mir_eval.separation.bss_eval_sources(
        baselines, estimates, compute_permutation=True)
    results = {
        'name': sample['name'],
        'sdr': sdr.tolist(),
        'sir': sir.tolist(),
        'sar': sar.tolist(),
        'nsdr': (sdr - baseline_sdr).tolist(),
    }
    return results


def write_results_to_json(sample, separator, model_name):
    results = estimate_and_evaluate(sample, separator, offset=10, duration=90)
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
    separator = tf.keras.models.load_model(model_path)
    # load the whole dsd100 dataset
    train_samples = create_samples('Dev')
    test_samples = create_samples('Test')
    eval_samples = test_samples + train_samples
    # computing metrics
    print('\nGenerating evaluation metrics on dsd100 samples...')
    for i in range(100):
        try:
            write_results_to_json(eval_samples[i], separator, model_name=model_name)
        except (ValueError):
            print(eval_samples[i]['name'] + ' is skipped')
            continue


if __name__ == '__main__':
    main('conv_res56_denoising_unet?time=20200227_0646_l2_reg.h5')
    # main('conv_encoder_denoising_decoder?time=20200227_0838_l2_weight_regularization.h5')
    # main('conv_denoising_unet?time=20200223_0347.h5')
