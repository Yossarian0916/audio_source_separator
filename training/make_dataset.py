import os
from utils.helper import get_data_dir_filenames
from utils.dataset import tfrecord2dataset


class DSD100Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_tfrecords = None
        self.valid_tfrecords = None
        self.test_tfrecords = None
        self.train_data_size = None
        self.valid_data_size = None
        self.test_data_size = None

    def get_datasets(self):
        self.build_datasets()
        return (self.train_dataset, self.valid_dataset, self.test_dataset)

    def build_datasets(self):
        dsd100_train_tfrecords, dsd100_test_tfrecords = self.get_tfrecords()
        # define tfrecords for datasets
        self.train_tfrecords = dsd100_train_tfrecords
        self.valid_tfrecords = dsd100_test_tfrecords[:len(dsd100_test_tfrecords)//2]
        self.test_tfrecords = dsd100_test_tfrecords
        # build datasets
        self.build_train_dataset(self.train_tfrecords)
        self.build_valid_dataset(self.valid_tfrecords)
        self.build_test_dataset(self.test_tfrecords)

    def dataset_stat(self):
        # train data samples
        if self.train_dataset is None:
            self.train_data_size = 0
        else:
            self.train_data_size = len(self.train_tfrecords)
        # valid data samples
        if self.valid_dataset is None:
            self.valid_data_size = 0
        else:
            self.valid_data_size = len(self.valid_tfrecords)
        # test data samples
        if self.test_dataset is None:
            self.test_data_size = 0
        else:
            self.test_data_size = len(self.test_tfrecords)
        return self.train_data_size, self.valid_data_size, self.test_data_size

    def build_train_dataset(self, train_tfrecords):
        if train_tfrecords is not None:
            self.train_dataset = tfrecord2dataset(train_tfrecords, self.batch_size)

    def build_valid_dataset(self, valid_tfrecords):
        if valid_tfrecords is not None:
            self.valid_dataset = tfrecord2dataset(valid_tfrecords, self.batch_size)

    def build_test_dataset(self, test_tfrecords):
        if test_tfrecords is not None:
            self.test_dataset = tfrecord2dataset(test_tfrecords, self.batch_size, shuffle=False)

    def get_tfrecords(self):
        dsd100_train_tfrecords = get_data_dir_filenames('dsd100_train_stft'+'/*')
        dsd100_test_tfrecords = get_data_dir_filenames('dsd100_test_stft'+'/*')
        return dsd100_train_tfrecords, dsd100_test_tfrecords
