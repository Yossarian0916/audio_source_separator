import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class UnetAutoencoder:
    def __init__(self, freq_bins, time_frames):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None

    def conv1d_bn(self, filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=None):
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

    def autoencoder(self, kernel_size, strides=1, name='autoencoder'):
        inputs = keras.Input(shape=(self.bins, self.frames))
        x = keras.layers.Conv1D(self.frames, kernel_size,
                                strides, padding='same', activation='relu')(inputs)
        x = keras.layers.MaxPool1D(2, padding='same')(x)
        x = keras.layers.Conv1D(self.frames, kernel_size,
                                strides, padding='same', activation='relu')(x)
        x = keras.layers.MaxPool1D(2, padding='same')(x)
        x = keras.layers.Conv1D(self.frames, kernel_size,
                                strides, padding='same', activation='relu')(x)
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Conv1D(self.frames, kernel_size, strides,
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Conv1D(self.frames, kernel_size,
                                strides, padding='same', activation='relu')(x)

        output = keras.layers.Conv1D(
            self.frames, 4, strides, padding='valid', activation='relu')(x)
        return keras.Model(inputs=[inputs], outputs=[output], name=name)

    def get_model(self, name='unet_dae_separator'):
        # input
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')

        # encoder
        conv1 = self.conv1d_bn(self.frames, 15)(mix_input)
        maxpool1 = keras.layers.MaxPool1D(2, padding='same')(conv1)
        conv2 = self.conv1d_bn(self.frames, 15)(maxpool1)
        maxpool2 = keras.layers.MaxPool1D(2, padding='same')(conv2)
        conv3 = self.conv1d_bn(self.frames, 15)(maxpool2)
        maxpool3 = keras.layers.MaxPool1D(2, padding='same')(conv3)
        conv4 = self.conv1d_bn(self.frames, 15)(maxpool3)

        # decoder
        conv4_upsampling = keras.layers.UpSampling1D(2)(conv4)
        conv5_input = self.crop_and_concat(conv4_upsampling, conv3)
        conv5 = self.conv1d_bn(self.frames, 15, padding='same')(conv5_input)

        conv5_upsampling = keras.layers.UpSampling1D(2)(conv5)
        conv6_input = self.crop_and_concat(conv5_upsampling, conv2)
        conv6 = self.conv1d_bn(self.frames, 15, padding='same')(conv6_input)
        conv6_upsampling = keras.layers.UpSampling1D(2)(conv6)

        conv7_input = self.crop_and_concat(conv6_upsampling, conv1)
        conv7 = self.conv1d_bn(self.frames, 15, padding='same')(conv7_input)

        conv8 = self.conv1d_bn(self.frames*3, 15, padding='same')(conv7)
        output = self.crop_and_concat(conv8, mix_input)

        # denoising autoencoder separators
        vocals = self.autoencoder(15, name='vocals')(
            output[:, :, :self.frames])
        bass = self.autoencoder(15, name='bass')(
            output[:, :, self.frames:self.frames*2])
        drums = self.autoencoder(15, name='drums')(
            output[:, :, self.frames*2:self.frames*3])
        other = self.autoencoder(15, name='other')(
            output[:, :, self.frames*3:])

        self.model = keras.Model(inputs=[mix_input],
                                 outputs=[vocals, bass, drums, other],
                                 name=name)
        return self.model

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def model_summary(self):
        if self.model is not None:
            trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
            non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])
            self.summary = {
                'total_parameters': trainable_count + non_trainable_count,
                'trainable_parameters': trainable_count,
                'non_trainable_parameters': non_trainable_count}
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

    def __str__(self):
        return self.summary

    __repr__ = __str__
