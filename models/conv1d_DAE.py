import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Conv1dDAE:
    def __init__(self, freq_bins, time_frames):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None

    def crop_and_concat(self, x1, x2):
        """crop tensor x1 to match x2, x2 shape is the target shape"""
        if x2 is None:
            return x1
        x1 = self.crop(x1, x2.get_shape().as_list())
        return tf.concat([x1, x2], axis=2)

    def crop(self, tensor, target_shape):
        """
        crop tensor to match target_shape,
        remove the diff/2 items at the start and at the end,
        keep only the central part of the vector
        """
        # the tensor flow in model is of shape (batch, freq_bins, time_frames)
        shape = tensor.get_shape().as_list()
        diff = shape[1] - target_shape[1]
        assert diff >= 0  # Only positive difference allowed
        if diff == 0:
            return tensor
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:, crop_start:-crop_end, :]

    def conv1d(self,
               input_tensor,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=None):
        x = keras.layers.Conv1D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                data_format='channels_last',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=-1)(x, training=True)
        output = keras.layers.LeakyReLU(0.01)(x)
        return output

    def autoencoder(self, kernel_size, name=None):
        """
        conv1d autoencoder, 
        treat spectrogram (time frames, frequency bins) as (sequence_length, feature_dimension),
        that means each time sequence has length equals to spectrogram temporal frames,
        each frequency bin is a feature dimension, similar to channel numbers in conv2d situation
        """
        transposed_spectrogram = keras.Input(shape=(self.frames, self.bins))
        x = self.conv1d(transposed_spectrogram, 32, 1)
        # downsampling
        x = self.conv1d(transposed_spectrogram, 64, kernel_size)
        x = keras.layers.MaxPool1D(2)(x)
        x = self.conv1d(x, 32, kernel_size)
        x = keras.layers.MaxPool1D(2)(x)
        # intermiate low dimension features
        x = self.conv1d(x, 32, kernel_size)
        # upsampling
        x = keras.layers.UpSampling1D(2)(x)
        x = self.conv1d(x, 32, kernel_size)
        x = keras.layers.UpSampling1D(2)(x)
        x = self.conv1d(x, 64, kernel_size)

        reconstructed = self.conv1d(x, self.bins, 1)
        reshaped = keras.layers.Reshape((self.bins, self.frames))(reconstructed)
        return keras.Model(inputs=[transposed_spectrogram], outputs=[reshaped], name=name)

    def get_model(self, name='conv1d_denoising_autoencoder'):
        # input
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        transposed_spectrogram = keras.layers.Reshape((self.frames, self.bins))(spectrogram)

        # denoising autoencoders, each autoencoder to separate one stem track
        vocals = self.autoencoder(kernel_size=3, name='vocals')(transposed_spectrogram)
        bass = self.autoencoder(kernel_size=3, name='bass')(transposed_spectrogram)
        drums = self.autoencoder(kernel_size=3, name='drums')(transposed_spectrogram)
        other = keras.layers.Subtract(name='other')([spectrogram, (vocals + bass + drums)])

        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        raise NotImplementedError

    def load_weights(self, path):
        raise NotImplementedError

    def save_model_plot(self, file_name='conv1d_denoising_autoencoder.png'):
        if self.model is not None:
            root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            images_dir = os.path.join(root_dir, 'images')
            file_path = os.path.join(images_dir, file_name)
            keras.utils.plot_model(self.model, file_path)
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

    def model_summary(self):
        if self.model is not None:
            trainable_count = np.sum([keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable_count = np.sum([keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            self.summary = {
                'total_parameters': trainable_count + non_trainable_count,
                'trainable_parameters': trainable_count,
                'non_trainable_parameters': non_trainable_count}
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

    def __str__(self):
        return self.summary

    __repr__ = __str__
