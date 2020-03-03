import functools
import multiprocessing as mp
import json
import os
import sys
import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from utils import module_path
from utils.dataset import create_samples
from utils.helper import wav_to_spectrogram_clips, rebuild_audio_from_spectro_clips
from evaluation import metrics


def load_model(path):
    """load pre-trained model"""
    try:
        saved_model_path = module_path.get_saved_model_path()
        pre_trained_models = os.listdir(saved_model_path)
        if path in pre_trained_models:
            separator = tf.keras.models.load_model(path)
        else:
            raise FileNotFoundError('no such pre-trained model .h5 file')
    except (OSError, TypeError):
        print('please check model file name!')
    return separator


def get_separated_tracks(separator, mix_audio):
    """return a list of separated stem components tracks"""
    # load mix music audio, average the stereo recording to single channel audio track
    # convert to spectrogram
    spectrogram = wav_to_spectrogram_clips(mix_audio)
    # separate components from the mix single channel music audio
    separated_spectrograms = separator.predict(spectrogram)
    # separated_spectrograms contains 4 stem tracks
    # the index of spectrograms: 0, 1, 2, 3 -> vocals, bass, drums, other
    separated_tracks = list()
    for i in range(4):
        separated_track = np.squeeze(separated_spectrograms[i], axis=-1)
        separated_tracks.append(rebuild_audio_from_spectro_clips(separated_track))
    return separated_tracks


def get_reference_tracks(sample, track_shape):
    """return a list of ground truth reference tracks of 4 stem components"""
    reference_tracks = list()
    for feat in ['vocals', 'bass', 'drums', 'other']:
        track, sr = librosa.load(sample[feat], sr=44100, mono=True)
        # crop reference track to match separated track shape
        track = track[tuple(map(slice, track_shape))]
        reference_tracks.append(track)
    return reference_tracks


def get_normalization_baseline(sample, track_shape):
    """return a list of ['mix', 'mix', 'mix', 'mix'] as normalization base"""
    track, sr = librosa.load(sample['mix'], sr=44100, mono=True)
    # crop reference track to match separated track shape
    track = track[tuple(map(slice, track_shape))]
    baseline_tracks = [track] * 4
    return baseline_tracks


def estimate_and_evaluate(sample, separator_model):
    # separation tracks
    # estimates = {
    #     'vocals': separated_tracks[0],
    #     'bass': separated_tracks[1],
    #     'drums': separated_tracks[2],
    #     'other': separated_tracks[3]
    # }
    separated_tracks = get_separated_tracks(separator_model, sample['mix'])
    estimates = np.asarray(separated_tracks)
    # reference tracks
    # references = {
    #     'vocals': reference_tracks[0],
    #     'bass': reference_tracks[1],
    #     'drums': reference_tracks[2],
    #     'other': reference_tracks[3],
    # }
    track_length = separated_tracks[0].shape
    reference_tracks = get_reference_tracks(sample, track_length)
    for i in range(4):
        assert reference_tracks[i].shape == separated_tracks[i].shape
    references = np.asarray(reference_tracks)
    # run bss-eval
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
        references, estimates, compute_permutation=True)
    # baselines bss_eval metrics
    # between ['vocals', 'bass', 'drums', 'other'] and mixture ['mix', 'mix', 'mix', 'mix]
    # these metrics act as normalization baselines
    baselines = get_normalization_baseline(sample, track_length)
    baselines = np.asarray(baselines)
    (baseline_sdr, baseline_sir, baseline_sar, _) = mir_eval.separation.bss_eval_sources(
        baselines, estimates, compute_permutation=True)
    # numpy.ndarray is not josn serializable, needs to convert into python list
    results = {
        'name': sample['name'],
        'sdr': sdr.tolist(),
        'sir': sir.tolist(),
        'sar': sar.tolist(),
        'nsdr': (sdr - baseline_sdr).tolist(),
    }
    return results


def write_results_to_json(sample, separator, model_name):
    results = estimate_and_evaluate(sample, separator)
    # release tensorflow model occupied memory
    tf.keras.backend.clear_session()
    # save evaluation results to .json file
    save_file_name = '_'.join(['eval', sample['name']]) + '.json'
    evaluation_path = module_path.get_evaluation_path()
    save_results_path = os.path.join(evaluation_path, model_name)
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    results_file = os.path.join(evaluation_path, model_name, save_file_name)
    with open(results_file, 'w', encoding='utf-8') as fd:
        json.dump(results, fd, sort_keys=True, indent=4)


def main(pre_trained_model_path):
    # load pre-trained model
    separator = load_model(pre_trained_model_path)
    if pre_trained_model_path == '':
        raise AttributeError('the model path is empty!')
    else:
        _, model_name = os.path.split(pre_trained_model_path)
    write_results_to_json_partial = functools.partial(
        write_results_to_json, model=separator, model_name=model_name)
    # load the whole dsd100 dataset
    train_samples = create_samples('Dev')
    test_samples = create_samples('Test')
    evaluation_samples = train_samples + test_samples
    # parallel computing
    num_cores = os.cpu_count()
    pool = mp.Pool(processes=num_cores)
    result = pool.map_async(write_results_to_json_partial, evaluation_samples)
    pool.close()
    pool.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='bss_eval separation quality')
    parser.add_argument('-m', '--model',
                        dest='model_path',
                        action='store',
                        type=str,
                        help='pre-trained neural network separator model .h5 file path')
    args = parser.parse_args()

    main(pre_trained_model_path='conv_denoising_unet?time=20200226_1546_with_sum_constraint.h5')
