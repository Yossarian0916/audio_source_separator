import tensorflow as tf
from functools import partial
import multiprocessing as mp
import os
import sys
from utils.helper import wav_to_spectrogram_clips, wav_to_log_spectrogram_clips
from utils import module_path


# config parameters
feat_names = ['mix', 'vocals', 'bass', 'drums', 'other']
cpu_cores = mp.cpu_count()


def get_filepath(basename, stem_type, dataset_name):
    data_dir = module_path.get_data_path()
    filepath = os.path.join(data_dir, dataset_name, stem_type, basename)
    return filepath


def tfrecord2dataset(filenames,
                     batch_size,
                     shuffle=True,
                     repeat_epochs=None,
                     n_readers=cpu_cores,
                     n_parse_threads=cpu_cores):
    dataset = tf.data.TFRecordDataset(filenames)
    # concurrently parse each tfrecord
    dataset = dataset.map(parse_records, num_parallel_calls=n_parse_threads)
    # caches elements in local file
    cache_file = os.path.join(module_path.get_data_path(), 'model_training_dataset_cache')
    dataset = dataset.cache(cache_file)
    # for perfect shuffling, the buffer size should be greater than the full size of the dataset
    if shuffle:
        dataset.shuffle(len(filenames))
    # repeat the dataset endless times
    dataset = dataset.repeat(repeat_epochs)
    # create batches
    dataset = dataset.batch(batch_size)
    return dataset


def parse_records(serialized_example, feat_names=feat_names):
    feat_description = {key: tf.io.FixedLenSequenceFeature(
        [], dtype=tf.float32, allow_missing=True) for key in feat_names}
    feat_description['freq_channels'] = tf.io.FixedLenFeature([], tf.int64)
    feat_description['time_frames'] = tf.io.FixedLenFeature([], tf.int64)
    sample = tf.io.parse_single_example(serialized_example, feat_description)
    # reshape each flattened array to 2D array
    freq_channels = tf.cast(sample['freq_channels'], tf.int64)
    time_frames = tf.cast(sample['time_frames'], tf.int64)
    for key in feat_names:
        sample[key] = tf.reshape(
            sample[key], tf.stack([freq_channels, time_frames]))
    return ({'mix': sample['mix']},
            {'vocals': sample['vocals'], 'bass': sample['bass'], 'drums': sample['drums'], 'other': sample['other']})


def write_records(sample, basename, transform_fn=wav_to_spectrogram_clips, feat_names=feat_names, compression_type=None):
    # tfrecord compression type
    tfrecord_opt = tf.io.TFRecordOptions(compression_type=compression_type)

    # data-structure used here
    # sample = {'name': string,
    #           'mix': wav file,
    #           'vocals': wav file,
    #           'bass': wav file,
    #           'drums': wav file,
    #           'other': wav file}
    #
    # each wav audio is chopped into 2-second-long clips, thus 87 frames
    # audio_clips = {'mix': (None, 2049, 87),
    #                'vocals': (None, 2049, 87),
    #                'bass': (None, 2049, 87),
    #                'drums': (None, 2049, 87),
    #                'other':(None, 2049, 87)}
    audio_clips = {key: None for key in feat_names}
    for feat in feat_names:
        audio_clips[feat] = transform_fn(sample[feat])
    # number of music clips after chopping each audio into 2-second-long sections
    clip_num = len(audio_clips['mix'])

    # audio_tracks = {'mix': (2049, 87),
    #                'vocals': (2049, 87),
    #                'bass': (2049, 87),
    #                'drums': (2049, 87),
    #                'other':(2049, 87)}
    for clip_id in range(clip_num):
        audio_tracks = {key: None for key in feat_names}
        for feat in feat_names:
            audio_tracks[feat] = audio_clips[feat][clip_id]
        filename_fullpath = '{}_song{}_{:03d}-of-{:03d}'.format(
            basename, sample['name'][:3], clip_id, clip_num)
        with tf.io.TFRecordWriter(filename_fullpath, tfrecord_opt) as writer:
            writer.write(serialize_example(audio_tracks))


def _float_feature(value):
    """returns a float_list from a float/double"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    """returns an int64_list from a bool/enum/int/unit"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(audio_tracks, feat_names=feat_names):
    # get shape of each 2D array, because tf.train.FloatList only accepts 'flat' list
    freq_channels = audio_tracks['mix'].shape[0]
    time_frames = audio_tracks['mix'].shape[1]
    # create example
    feature = {key: _float_feature(audio_tracks[key]) for key in feat_names}
    feature['freq_channels'] = _int64_feature(freq_channels)
    feature['time_frames'] = _int64_feature(time_frames)
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def create_samples(basename='Dev', dataset_name='DSD100', feat_names=feat_names):
    mixture_songs_filepath = get_filepath(basename, 'Mixtures', dataset_name)
    stem_sources_filepath = get_filepath(basename, 'Sources', dataset_name)
    song_names = sorted(os.listdir(mixture_songs_filepath))
    # store all samples, eache sample is a hash table
    # sample = {'name': string,
    #           'mix': wav file,
    #           'vocals': wav file,
    #           'bass': wav file,
    #           'drums': wav file,
    #           'other': wav file}
    samples = list()
    for song in song_names:
        sample = dict.fromkeys(['name']+feat_names, None)
        mixture_song_filepath = os.path.join(mixture_songs_filepath, song)
        song_sources_filepath = os.path.join(stem_sources_filepath, song)
        sample['name'] = song
        sample['mix'] = os.path.join(mixture_song_filepath, 'mixture.wav')
        sample['vocals'] = os.path.join(song_sources_filepath, 'vocals.wav')
        sample['bass'] = os.path.join(song_sources_filepath, 'bass.wav')
        sample['drums'] = os.path.join(song_sources_filepath, 'drums.wav')
        sample['other'] = os.path.join(song_sources_filepath, 'other.wav')
        samples.append(sample)
    return samples


def generate_tfrecords_files(usage, transform):
    # get data wav files
    data_dir = module_path.get_data_path()
    # create one audio sample for training and test
    if usage == 'train':
        samples = create_samples(basename='Dev')
    elif usage == 'test':
        samples = create_samples(basename='Test')
    else:
        raise ValueError("usage should be of type string, value is 'train' or 'test'")
    # define signal transformation to be applied on audio samples
    if transform == 'stft':
        transform_fn = wav_to_spectrogram_clips
    elif transform == 'logstft':
        transform_fn = wav_to_log_spectrogram_clips
    else:
        raise ValueError("transform_fn should be of type string, value is 'stft' or 'logstft'")
    # define output dataset directory
    output_dir = os.path.join(data_dir, 'dsd100_{}_{}'.format(usage, transform))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('output path: ', output_dir,  '\n......')
    basename = os.path.join(output_dir, usage)
    # generate tfrecord file in parallel
    write_records_partial = partial(
        write_records, basename=basename, transform_fn=transform_fn)
    # use process pool to create tfrecords files
    num_cores = os.cpu_count()
    pool = mp.Pool(processes=num_cores)
    result = pool.map_async(write_records_partial, samples)
    pool.close()
    pool.join()
