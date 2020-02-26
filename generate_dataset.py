#!/usr/bin/env python3

import os
import sys
import argparse
from utils.dataset import generate_tfrecords_files


def main():
    parser = argparse.ArgumentParser(description='Generate dataset tfrecord files')
    parser.add_argument('-t', '--transform_method',
                        dest='transform',
                        action='store',
                        default='stft',
                        type=str,
                        choices=['stft', 'logstft'],
                        help='signal transform is to apply on waveform audio data')

    args = parser.parse_args()
    print('\nGenerating tfrecords:')
    # generate dataset tfrecord files with given transformation function
    print('\ngenerating training data samples with', args.audio_transform, '...')
    generate_tfrecords_files('train', str(args.audio_transform))
    print('\ngenerating test data samples with', args.audio_transform, '...')
    generate_tfrecords_files('test', str(args.audio_transform))
    print('Done!')


if __name__ == '__main__':
    main()
