import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import models.util as util


class Conv2dResblockDenoisingUnet:
    def __init__(self, freq_bins, time_frames, kernel_size=(3, 3)):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None
        self.kernel_size = kernel_size

    def identity_block(self,
                       input_tensor,
                       filters,
                       kernel_size,
                       kernel_initializer='he_normal',
                       kernel_regularizer=None):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1), padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size, padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter3, (1, 1), padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)
        output = keras.layers.Add()([x, input_tensor])
        return output

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   kernel_initializer='he_normal',
                   kernel_regularizer=None):
        """residual block with convolutional skip connection, used to change input tensor channels"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1), strides=strides, padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size, padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x, training=True)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter3, (1, 1), padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)
        # skip connection
        shortcut = keras.layers.Conv2D(filter3, (1, 1), strides=strides, padding='same',
                                       kernel_initializer=kernel_initializer,
                                       kernel_regularizer=kernel_regularizer, name='projection')(input_tensor)
        shortcut = keras.layers.BatchNormalization(axis=bn_axis)(shortcut, training=True)
        output = keras.layers.Add()([x, shortcut])
        return output

    def resblock_unet(self):
        spectrogram = keras.Input(shape=(self.bins, self.frames, 1))

        conv = keras.layers.Conv2D(4, (1, 1), padding='same', activation='relu')(spectrogram)
        bn = keras.layers.BatchNormalization()(conv)

        filters_set = [[4 << i, 4 << i, 8 << i] for i in range(4)]
        # encoder
        # Conv1 + residual identity block + downsampling
        conv1 = self.conv_block(bn, filters_set[0], self.kernel_size, strides=(2, 2))
        res_block1 = self.identity_block(conv1, filters_set[0], self.kernel_size)

        # Conv2 + residual identity block + maxpooling
        conv2 = self.conv_block(res_block1, filters_set[1], self.kernel_size, strides=(2, 2))
        res_block2 = self.identity_block(conv2, filters_set[1], self.kernel_size)

        # Conv3 + residual identity block + maxpooling
        conv3 = self.conv_block(res_block2, filters_set[2], self.kernel_size, strides=(2, 2))
        res_block3 = self.identity_block(conv3, filters_set[2], self.kernel_size)

        # latent tensor, compressed features
        # Conv4 + residual identity block
        conv4 = self.conv_block(res_block3, filters_set[3], self.kernel_size)
        res_block4 = self.identity_block(conv4, filters_set[3], self.kernel_size)

        # decoder
        # upsampling + residual identity block + Conv5
        upsample1 = keras.layers.UpSampling2D((2, 2))(res_block4)
        conv5 = self.conv_block(upsample1, filters_set[2], self.kernel_size)
        res_block5 = self.identity_block(util.crop_and_concat(conv5, res_block3), filters_set[2], self.kernel_size)

        # upsampling + residual identity block + Conv6
        upsample2 = keras.layers.UpSampling2D((2, 2))(res_block5)
        conv6 = self.conv_block(upsample2, filters_set[1], self.kernel_size)
        res_block6 = self.identity_block(util.crop_and_concat(conv6, res_block2), filters_set[1], self.kernel_size)

        # upsampling + residual identity block + Conv7
        upsample3 = keras.layers.UpSampling2D((2, 2))(res_block6)
        conv7 = self.conv_block(upsample3, filters_set[0], self.kernel_size)
        res_block7 = self.identity_block(util.crop_and_concat(conv7, res_block1), filters_set[0], self.kernel_size)

        # output layers
        conv8 = self.conv_block(res_block7, [4, 4, 1], self.kernel_size)
        res_block8 = self.identity_block(conv8, [4, 4, 1], self.kernel_size)
        res_block9 = self.identity_block(conv8, [4, 4, 1], self.kernel_size)
        return keras.Model(inputs=[spectrogram], output=[res_block9])

    def get_model(self, name='conv_spectrogram_resnet_autoencoder'):
        """autoencoder with resnet as encoder"""
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        vocals = self.resblock_unet(reshaped_spectrogram, name='vocals')
        bass = self.resblock_unet(reshaped_spectrogram, name='bass')
        drums = self.resblock_unet(reshaped_spectrogram, name='drums')
        other = self.resblock_unet(reshaped_spectrogram, name='other')
        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        raise NotImplementedError

    def load_weights(self, path):
        raise NotImplementedError

    def save_model_plot(self, file_name='conv2d_resnet_autoencoder_separator.png'):
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
