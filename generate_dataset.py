#!/usr/bin/env python3

import os
import sys
import argparse
from utils.dataset import generate_tfrecords_files


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser(
            description='Generate dataset tfrecord files')
        parser.add_argument('usage', type=str, choices=['train', 'test'],
                            help='the usage of dataset')
        parser.add_argument('audio_transform', type=str, choices=['stft', 'logstft'],
                            help='decide which kind transformation is to applied on wav audio signals')

        # pass arguments to generate dataset tfrecords
        args = parser.parse_args()
        print('Generating tfrecords: training data samples with',
              args.audio_transform, '...')
        generate_tfrecords_files(str(args.usage), str(args.audio_transform))

    main()
