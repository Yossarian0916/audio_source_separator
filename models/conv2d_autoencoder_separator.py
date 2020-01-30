import tensorflow as tf
from tensorflow import keras


class AutoencoderConv2d:
    def __init__(self, freq_bins, time_frames):
        self.bins = freq_bins
        self.frames = time_frames

    def get_model(self):
        """FCN design, autoencoder with concat skip connection"""
        mix_input = keras.Input(shape=(self.frames, self.bins, 1), name='mix')
        # encoder
        # 1st conv + downsampling + batch normalization
        conv1 = keras.layers.Conv2D(32, (31, 3), padding='same', activation='relu')(mix_input)
        downsample1 = keras.layers.Conv2D(32, (31, 3), strides=(2, 2), padding='same',
                                          activation='relu', use_bias=False)(conv1)
        x = keras.layers.BatchNormalization()(downsample1)
        # 2nd conv + downsampling + batch normalization
        conv2 = keras.layers.Conv2D(64, (31, 3), padding='same', activation='relu')(x)
        downsample1 = keras.layers.Conv2D(64, (31, 3), strides=(2, 2), padding='same',
                                          activation='relu', use_bias=False)(conv2)
        x = BatchNormalization()(downsample1)
        # 3rd conv + batch normalization
        conv3 = keras.layers.Conv2D(128, (31, 3), padding='same', activation='relu', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(conv3)
        # decoder
        # 4th conv + upsampling + batch normalization
        upsample1 = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (2, 2), activation='relu', use_bias=False)(x)
        x = keras.layers.Concatenate([conv2, upsample1])
        conv4 = keras.layers.Conv2D(64, (31, 3), padding='same', activatoin='relu')(x)
        x = keras.layers.BatchNormalization()(conv4)
        # 5th conv + upsampling + batch normalization
        upsample2 = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (2, 2), activation='relu', use_bias=False)(x)
        x = keras.layers.Concatenate([conv1, upsample2])
        conv5 = keras.layers.Conv2D(32, (31, 3), padding='same', activatoin='relu')(x)
        x = keras.layers.BatchNormalization()(conv5)
        # output layers
        x = keras.layers.Conv2D(32, (31, 3), padding='same', activation='relu')(x)
        x = keras.layers.Conv2D(16, (31, 3), padding='same', activation='relu')(x)
        output = keras.layers.Conv2D(4, (31, 3), padding='same', activation='relu')(x)

        vocals = output[:, :, :1]
        bass = output[:, :, 1:2]
        drums = output[:, :, 2:3]
        other = output[:, :, 3:]
        return vocals, bass, drums, other

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass
