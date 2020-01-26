import tensorflow as tf
from tensorflow import keras


class UNet_Autoencoder:
    def __init__(self, freq_bins, time_frames):
        self.bins = freq_bins
        self.frames = time_frames

    def conv1d_bn(self, filters, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal', name=None):
        conv1d_bn = keras.Sequential([
            keras.layers.Conv1D(filters, kernel_size, strides, padding,
                                activation=None, kernel_initializer=kernel_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.01),
        ], name=name)
        return conv1d_bn

    def crop_and_concat(self, x1, x2):
        if x2 is None:
            return x1
        x1 = self.crop(x1, x2.get_shape().as_list())
        return tf.concat([x1, x2], axis=2)

    def crop(self, tensor, target_shape):
        shape = tensor.get_shape().as_list()
        diff = shape[1] - target_shape[1]
        assert diff >= 0  # Only positive difference allowed
        if diff == 0:
            return tensor
        crop_start = diff // 2
        crop_end = diff - crop_start

        return tensor[:, crop_start:-crop_end, :]

    def autoencoder(self, kernel_size=1, strides=1, name='denoising_autoencoder'):
        track_stft = keras.Input(shape=(self.bins, self.frames))
        x = keras.layers.Conv1D(self.frames * 3, kernel_size,
                                strides, padding='same', activation='relu')(track_stft)
        x = keras.layers.MaxPool1D(pool_size=2, padding='same')(x)
        x = keras.layers.Conv1D(self.frames * 2, kernel_size,
                                strides, padding='same', activation='relu')(x)
        x = keras.layers.MaxPool1D(pool_size=2, padding='same')(x)
        x = keras.layers.Conv1D(self.frames, kernel_size,
                                strides, padding='same', activation='relu')(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(self.frames * 2, kernel_size=2,
                                strides=strides, padding='valid', activation='relu')(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        output = keras.layers.Conv1D(
            self.frames, kernel_size=2, strides=strides, padding='valid', activation='relu')(x)
        return keras.Model(inputs=[track_stft], outputs=[output], name=name)

    def get_model(self, name=None):
        # input
        mix_input = keras.Input(
            shape=(self.bins, self.frames), name='mix_input')

        # encoder
        conv1 = self.conv1d_bn(self.frames, 3, name='conv1')(mix_input)
        maxpool1 = keras.layers.MaxPool1D(
            2, padding='same', name='maxpool1')(conv1)
        conv2 = self.conv1d_bn(self.frames * 2, 3, name='conv2')(maxpool1)
        maxpool2 = keras.layers.MaxPool1D(
            2, padding='same', name='maxpool2')(conv2)
        conv3 = self.conv1d_bn(self.frames * 3, 3, name='conv3')(maxpool2)
        maxpool3 = keras.layers.MaxPool1D(
            pool_size=2, padding='same', name='maxpool3')(conv3)
        conv4 = self.conv1d_bn(self.frames * 4, 3, name='conv4')(maxpool3)

        # decoder
        conv4_upsampling = keras.layers.UpSampling1D(
            2, name='upsample1')(conv4)
        conv5_input = self.crop_and_concat(conv4_upsampling, conv3)
        conv5 = self.conv1d_bn(self.frames * 3, 3, padding='same',
                               name='conv5')(conv5_input)

        conv5_upsampling = keras.layers.UpSampling1D(
            2, name='upsample2')(conv5)
        conv6_input = self.crop_and_concat(conv5_upsampling, conv2)
        conv6 = self.conv1d_bn(self.frames * 2, 3, padding='same',
                               name='conv6')(conv6_input)
        conv6_upsampling = keras.layers.UpSampling1D(
            2, name='upsample3')(conv6)

        conv7_input = self.crop_and_concat(conv6_upsampling, conv1)
        conv7 = self.conv1d_bn(self.frames*4, 3, padding='same',
                               name='conv7')(conv7_input)

        # denoising autoencoder separators
        vocals = self.autoencoder(name='vocals')(
            conv7[:, :, :self.frames])
        bass = self.autoencoder(name='bass')(
            conv7[:, :, self.frames:self.frames*2])
        drums = self.autoencoder(name='drums')(
            conv7[:, :, self.frames*2:self.frames*3])
        other = self.autoencoder(name='other')(
            conv7[:, :, self.frames*3:])

        return keras.Model(inputs=[mix_input], outputs=[vocals, bass, drums, other], name=name)
