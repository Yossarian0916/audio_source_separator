import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import models.util as util


class ConvDenoisingUnet:
    def __init__(self, freq_bins, time_frames, kernel_size=(3, 3)):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None
        self.kernel_size = kernel_size

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   padding='same',
                   use_bias=True,
                   kernel_initializer='he_normal',
                   kernel_regularizer=None):
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                activation='relu',
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization()(x, training=True)
        return x

    def unet(self, name='unet'):
        """crop-and-concat features as skip connection, fully convolutional structure"""
        spectrogram = keras.Input(shape=(self.bins, self.frames, 1))

        conv0 = self.conv_block(spectrogram, 4, self.kernel_size)
        # encoder
        # 1st downsampling
        conv1 = self.conv_block(conv0, 8, self.kernel_size)
        downsample1 = self.conv_block(conv1, 8, self.kernel_size, strides=(2, 2))

        # 2nd downsampling
        conv2 = self.conv_block(downsample1, 16, self.kernel_size)
        downsample2 = self.conv_block(conv2, 16, self.kernel_size, strides=(2, 2))

        # intermediate low dimensional features
        conv3 = self.conv_block(downsample2, 32, self.kernel_size)
        conv4 = self.conv_block(conv3, 64, self.kernel_size)
        conv5 = self.conv_block(conv4, 32, self.kernel_size)

        # decoder
        # 4th upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(conv5)
        conv6 = self.conv_block(util.crop_and_concat(upsample1, conv2), 16, self.kernel_size)

        # 5th upsampling
        upsample2 = keras.layers.UpSampling2D((2, 2))(conv6)
        conv7 = self.conv_block(util.crop_and_concat(upsample2, conv1), 8, self.kernel_size)

        # output layers
        output = self.conv_block(conv7, 1, self.kernel_size)
        return keras.Model(inputs=[spectrogram], outputs=[output], name=name)

    def get_model(self, name='conv_denoising_unet'):
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        vocals = self.unet(name='vocals')(reshaped_spectrogram)
        bass = self.unet(name='bass')(reshaped_spectrogram)
        drums = self.unet(name='drums')(reshaped_spectrogram)
        other = keras.layers.Subtract(name='other')([reshaped_spectrogram, (vocals + bass + drums)])
        #other = self.unet(name='other')(reshaped_spectrogram)

        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def save_model_plot(self, file_name='conv_denoising_unet.png'):
        if self.model is not None:
            root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            images_dir = os.path.join(root_dir, 'images')
            if not os.path.exists(images_dir):
                os.mkdir(images_dir)
            file_path = os.path.join(images_dir, file_name)
            keras.utils.plot_model(self.model, file_path)
        else:
            raise ValueError(
                "no model has been built yet! call get_model() first!")

    def model_summary(self):
        if self.model is not None:
            trainable_count = np.sum([keras.backend.count_params(w)
                                      for w in self.model.trainable_weights])
            non_trainable_count = np.sum(
                [keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            self.summary = {
                'total_parameters': trainable_count + non_trainable_count,
                'trainable_parameters': trainable_count,
                'non_trainable_parameters': non_trainable_count}
        else:
            raise ValueError(
                "no model has been built yet! call get_model() first!")

    def __str__(self):
        return self.summary

    __repr__ = __str__
