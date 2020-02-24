import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import models.util as util


class ConvEncoderDenoisingDecoder:
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

    def encoder(self, input_tensor):
        x = self.conv_block(input_tensor, 4, self.kernel_size)
        # 1st downsampling
        conv1 = self.conv_block(x, 8, self.kernel_size)
        downsample1 = self.conv_block(conv1, 8, self.kernel_size, strides=(2, 2))
        # 2nd downsampling
        conv2 = self.conv_block(downsample1, 16, self.kernel_size)
        downsample2 = self.conv_block(conv2, 16, self.kernel_size, strides=(2, 2))
        # 3rd downsampling
        conv3 = self.conv_block(downsample1, 32, self.kernel_size)
        downsample3 = self.conv_block(conv3, 32, self.kernel_size, strides=(2, 2))
        # intermediate low dimensional features
        x = self.conv_block(downsample3, 32, self.kernel_size)
        x = self.conv_block(x, 64, self.kernel_size)
        latent_space = self.conv_block(x, 32, self.kernel_size)
        return latent_space, conv3, conv2, conv1

    def decoder(self, input_tensors, name=None):
        latent_space, conv3, conv2, conv1 = input_tensors
        # 1st upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(latent_space)
        x = self.conv_block(util.crop_and_concat(upsample1, conv3), 16, self.kernel_size)
        # 2nd upsampling
        upsample2 = keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_block(util.crop_and_concat(upsample2, conv2), 8, self.kernel_size)
        # 3rd upsampling
        upsample3 = keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_block(util.crop_and_concat(upsample3, conv1), 4, self.kernel_size)
        output = keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', name=name)(x)
        return output

    def get_model(self, name='conv_denoising_unet'):
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        (latent, conv3, conv2, conv1) = self.encoder(reshaped_spectrogram)
        vocals = self.decoder((latent, conv3, conv2, conv1), name='vocals')
        bass = self.decoder((latent, conv3, conv2, conv1), name='bass')
        drums = self.decoder((latent, conv3, conv2, conv1), name='drums')
        other = self.decoder((latent, conv3, conv2, conv1), name='other')

        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def save_model_plot(self, file_name='conv_encoder_denoising_decoder.png'):
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
