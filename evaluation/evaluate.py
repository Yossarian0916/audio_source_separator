import multiprocessing as mp
import json
import os
import sys
import librosa
import mir_eval
import tensorflow as tf
from utils import module_path
from utils.dataset import create_sample


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
    mix_mono_audio, sr = librosa.load(mix_audio, mono=True, sr=44100)
    # separate components from the mix single channel music audio
    separated_tracks = separator.predict(mix_mono_audio)
    separated_vocals = separated_tracks[0]
    separated_bass = separated_tracks[1]
    separated_drums = separated_tracks[2]
    separated_other = separated_tracks[3]
    return [separated_vocals, separated_bass, separated_drums, separated_other]


def get_reference_tracks(sample):
    reference_tracks = list()
    for feat in ['vocals', 'bass', 'drums', 'other']:
        track, sr = librosa.load(sample[feat], mono=True, sr=44100)
        reference_tracks.append(track)
    return reference_tracks


def estimate_and_evaluate(sample, separator_model):
    # separation tracks
    separated_tracks = get_separated_tracks(separator_model, sample['mix'])
    estimates = {
        'vocals': separated_tracks[0],
        'bass': separated_tracks[1],
        'drums': separated_tracks[2],
        'other': separated_tracks[3]
    }
    # reference tracks
    reference_tracks = get_reference_tracks(sample)
    references = {
        'vocals': reference_tracks[0],
        'bass': reference_tracks[1],
        'drums': reference_tracks[2],
        'other': reference_tracks[3],
    }
    # run bss-eval
    evaluation_path = module_path.get_evaluation_path()
    results = {
        'name': sample['name'],
    }
    for feat in ['vocals', 'bass', 'drums', 'other']:
        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(references[feat], estimates[feat])
        results[feat] = {
            'sdr': sdr,
            'sir': sir,
            'sar': sar,
        }
    return results


def main(pre_trained_model_path):
    # load pre-trained model
    separator = load_model(pre_trained_model_path)
    # load the whole dsd100 dataset
    train_samples = create_sample('Dev')
    test_samples = create_sample('Test')
    evaluation_samples = train_samples + test_samples
    for sample in evaluation_samples:
        results = estimate_and_evaluate(sample, separator)
        # release tensorflow model occupied memory
        tf.keras.backend.clear_session()
        # save evaluation results to .json file
        save_json_file = '_'.join(['eval', sample['name']]) + '.json'
        with open(save_json_file, 'w') as fd:
            json.dump(results, fd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='bss_eval separation quality')
    parser.add_argument('-m', '--model',
                        dest='model_path',
                        action='store',
                        type=str,
                        help='pre-trained neural network separator model .h5 file path')
    args = parser.parse_args()
