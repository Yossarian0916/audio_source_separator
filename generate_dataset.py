#!/usr/bin/env python3

import os
import sys
import argparse
from utils.dataset import generate_tfrecords_files


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset tfrecord files')
    parser.add_argument('usage', type=str, choices=['train', 'test'],
                        help='the usage of dataset')
    parser.add_argument('audio_transform', type=str, choices=['stft', 'logstft'],
                        help='decide which kind transformation is to applied on wav audio signals')

    # pass arguments to generate dataset tfrecords
    args = parser.parse_args()
    if args.usage not in ['train', 'test']:
        raise argparse.ArgumentError(usage, 'should be of type string, value is train or test')
    if args.audio_transform not in ['stft', 'logstft']:
        raise argparse.ArgumentError(audio_transform, 'should be of type string, value is stft or logstft')
    print('\nGenerating tfrecords: ', args.usage, 'data samples with', args.audio_transform, '...')
    # generate dataset tfrecord files with given transformation function
    generate_tfrecords_files(str(args.usage), str(args.audio_transform))


if __name__ == '__main__':
    main()
    print('FINISHED')
