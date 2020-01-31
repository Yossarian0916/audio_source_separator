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
        """crop tensor x1 to match x2, x2 shape is the target shape"""
        if x2 is None:
            return x1
        x1 = self.crop(x1, x2.get_shape().as_list())
        return tf.concat([x1, x2], axis=2)

    def crop(self, tensor, target_shape):
        # the tensor flow in model is of shape (batch, freq_bins, time_frames)
        shape = tensor.get_shape().as_list()
        diff = shape[1] - target_shape[1]
        assert diff >= 0  # Only positive difference allowed
        if diff == 0:
            return tensor
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:, crop_start:-crop_end, :]

    def tile(self, tensor, multiples):
        freq_multiple, time_multiple = multiples
        # the tensor flow in model is of shape (batch, freq_bins, time_frames)
        multiples = tf.constant((1, freq_multiple, time_multiple), tf.int32)
        return tf.tile(tensor, multiples)

    def autoencoder(self, kernel_size, strides=1, name='autoencoder'):
        """denoising autoencoder to further remove noise"""
        inputs = keras.Input(shape=(self.bins, self.frames))
        # downsampling
        x = keras.layers.Conv1D(self.frames, kernel_size, strides=2, padding='same', activation='relu')(inputs)
        x = keras.layers.Conv1D(self.frames, kernel_size, strides=2, padding='same', activation='relu')(x)
        # upsampling
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Conv1D(self.frames, kernel_size, strides, padding='same', activation='relu')(x)
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Conv1D(self.frames, kernel_size, strides, padding='same', activation='relu')(x)
        # this skip connection is to resize the output layer tensor shape
        # instead of another conv1d layer, which kernel_size will be 4, too small
        # thus, will introduce new noisy pixels in spectrogram
        x = self.crop_and_concat(x, inputs)
        output = keras.layers.Conv1D(self.frames, kernel_size, strides, padding='same', activation='relu')(x)
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

        # output layer
        output = keras.layers.Add()([conv7, mix_input])
        output = self.tile(output, (1, 4))

        # uniformly split tensor along time_frames axis into 4 inputs
        vocals_input, bass_input, drums_input, other_input = tf.split(output, 4, axis=2)
        # denoising autoencoder separators
        vocals = self.autoencoder(15, name='vocals')(vocals_input)
        bass = self.autoencoder(15, name='bass')(bass_input)
        drums = self.autoencoder(3, name='drums')(drums_input)
        other = self.autoencoder(7, name='other')(other_input)

        self.model = keras.Model(inputs=[mix_input], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def save_model_plot(self, file_name='unet_dae_separator.png'):
        if self.model is not None:
            root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            images_dir = os.path.join(root_dir, 'images')
            file_path = os.path.join(images_dir, file_name)
            keras.utils.plot_model(self.model, file_path)
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

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
