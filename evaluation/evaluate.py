import os
import sys
import librosa
import mir_eval
import tensorflow as tf
from utils import module_path


def load_model(path):
    try:
        saved_model_path = module_path.get_saved_model_path()
        pre_trained_models = os.listdir(saved_model_path)
        separator = tf.keras.models.load_model(path)
    except (OSError, TypeError, AttributeError):
        print('please check model file name!')
    return separator


def get_separated_tracks(separator, audio_sample):
    audio_mono_track, sr = librosa.load(audio_sample, mono=True, sr=44100)
    separated_tracks = separator.predict(audio_mono_track)
    # separated components of the mix music audio
    separated_vocals = separated_tracks[0]
    separated_bass = separated_tracks[1]
    separated_drums = separated_tracks[2]
    separated_other = separated_tracks[3]
    return [separated_vocals, separated_bass, separated_drums, separated_other]


def estimate_and_evaluate(track, model_path):
    separator = load_model(model_path)
    separated_tracks = get_separated_tracks(separator, track)
    estimates = {
        'vocals': separated_tracks[0],
        'bass': separated_tracks[1],
        'drums': separated_tracks[2],
        'other': separated_tracks[3]
    }
    evaluation_path = module_path.get_evaluation_path()
    # run bss-eval
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='bss_eval separation quality')
    parser.add_argument('-s', '--sources',
                        dest='audio_sources_path',
                        action='store',
                        type=str,
                        help='mixed music audio and corresponding stem sources(vocal, bass, drums, other) files path')
    parser.add_argument('-m', '--model',
                        dest='model_path',
                        action='store',
                        type=str,
                        help='pre-trained neural network separator model .h5 file path')
    args = parser.parse_args()
