import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from layers.conv1D_layer import Conv1DTranspose


class Encoder(keras.layers.Layer):
    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.freq_bins = frequency_bins
        self.time_frames = time_frames
        self.conv1 = keras.layers.Conv1D(filters=self.freq_bins // 2,
                                         kernel_size=3,
                                         padding='same',
                                         activation='relu')
        self.conv2 = keras.layers.Conv1D(filters=self.freq_bins // 4,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu')
        self.conv3 = keras.layers.Conv1D(filters=self.freq_bins // 8,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         activation='relu')

    @tf.function
    def call(self, inputs):
        reshaped_inputs = keras.layers.Reshape(
            (self.time_frames, self.freq_bins))(inputs)
        conv1 = self.conv1(reshaped_inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv1, conv2, conv3


class Decoder(keras.layers.Layer):
    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.freq_bins = frequency_bins
        self.time_frames = time_frames
        self.tconv4 = Conv1DTranspose(
            filters=self.freq_bins // 4, kernel_size=3)
        self.tconv5 = Conv1DTranspose(
            filters=self.freq_bins // 2, kernel_size=3)
        self.tconv6 = Conv1DTranspose(filters=self.freq_bins, kernel_size=3)

    @tf.function
    def call(self, inputs):
        conv1, conv2, conv3 = inputs
        # 1st deconvolution layer with skip connection
        tconv4 = self.tconv4(conv3)
        tconv4_output = keras.layers.Add(name='skip_conn1')([tconv4, conv2])
        # 2nd deconvolution layer with skip connection
        tconv5 = self.tconv5(tconv4_output)
        tconv5_output = keras.layers.Add(name='skip_conn2')([tconv5, conv1])
        # output deconvolution layer
        tconv6 = self.tconv6(tconv5_output)
        output = keras.layers.Reshape(
            (self.freq_bins, self.time_frames))(tconv6)
        return output
