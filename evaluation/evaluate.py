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
    # load mix music audio, average the stereo recording to single channel audio track
    # convert to spectrogram
    spectrogram = wav_to_spectrogram_clips(mix_audio)
    # separate components from the mix single channel music audio
    separated_spectrograms = separator.predict(spectrogram)
    separated_tracks = list()
    # separated_spectrograms contains 4 stem tracks
    # the index of spectrograms: 0, 1, 2, 3 -> vocals, bass, drums, other
    for i in range(4):
        separated_track = np.squeeze(separated_spectrograms[i], axis=-1)
        separated_tracks.append(rebuild_audio_from_spectro_clips(separated_track))
    return separated_tracks


def get_reference_tracks(sample, track_shape):
    reference_tracks = list()
    for feat in ['vocals', 'bass', 'drums', 'other']:
        track, sr = librosa.load(sample[feat], sr=44100, mono=True)
        # crop reference track to match separated track shape
        track = track[tuple(map(slice, track_shape))]
        reference_tracks.append(track)
    return reference_tracks


def estimate_and_evaluate(sample, separator_model):
    # separation tracks
    separated_tracks = get_separated_tracks(separator_model, sample['mix'])
    estimates = np.asarray(separated_tracks)
    # estimates = {
    #     'vocals': separated_tracks[0],
    #     'bass': separated_tracks[1],
    #     'drums': separated_tracks[2],
    #     'other': separated_tracks[3]
    # }
    # reference tracks
    reference_tracks = get_reference_tracks(sample, separated_tracks[0].shape)
    for i in range(4):
        assert reference_tracks[i].shape == separated_tracks[i].shape
    #  references = {
    #      'vocals': reference_tracks[0],
    #      'bass': reference_tracks[1],
    #      'drums': reference_tracks[2],
    #      'other': reference_tracks[3],
    #  }
    references = np.asarray([reference_tracks[0],
                             reference_tracks[1],
                             reference_tracks[2],
                             reference_tracks[3]])
    # run bss-eval
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
        references, estimates, compute_permutation=False)
    results = {
        'name': sample['name'],
        'sdr': sdr,
        'sir': sir,
        'sar': sar
    }
    return results


def write_results_to_json(sample, separator):
    results = estimate_and_evaluate(sample, separator)
    # release tensorflow model occupied memory
    tf.keras.backend.clear_session()
    # save evaluation results to .json file
    save_file_name = '_'.join(['eval', sample['name']]) + '.json'
    evaluation_path = module_path.get_evaluation_path()
    json_file_path = os.path.join(evaluation_path, save_file_name)
    with open(json_file_path, 'w') as fd:
        json.dump(results, fd)


def main(pre_trained_model_path):
    # load pre-trained model
    separator = load_model(pre_trained_model_path)
    write_results_to_json_partial = functools.partial(
        write_results_to_json, model=separator)
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

    main(pre_trained_model_path='conv_resblock_denoising_unet?time=20200301_1113.h5')
