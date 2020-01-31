import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class DenoisingAutoencoder:
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

    def conv1d_bn(self,
                  input_tensor,
                  filters,
                  kernel_size,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(0.01)):
        x = keras.layers.Conv1D(filters, kernel_size,
                                strides=strides,
                                padding=padding,
                                data_format='channels_last',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        output = keras.layers.LeakyReLU(0.01)(x)
        return output

    def autoencoder(self, intput_tensor, kernel_size):
        """denoising autoencoder to further remove noise"""
        # downsampling
        x = self.conv1d_bn(input_tensor, self.frames, kernel_size, 2)
        x = self.conv1d_bn(x, self.frames, kernel_size, 2)
        # upsampling
        x = keras.layers.UpSampling1D(2)(x)
        x = self.conv1d_bn(x, self.frames, kernel_size)
        x = keras.layers.UpSampling1D(2)(x)
        x = self.conv1d_bn(x, self.frames, kernel_size)
        # this skip connection is to resize the output layer tensor shape
        # instead of another conv1d layer, which kernel_size should be 4 in order
        # to match the input tensor shape (frequency_bins, time_frames)
        # kernel_size=4 is too small, will introduce new noisy pixels in spectrogram
        x = self.crop_and_concat(x, input_tensor)
        output = self.conv1d_bn(x, self.frames, kernel_size)
        return output

    def get_model(self, name='denoising_autoencoder'):
        # input
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')
        # denoising autoencoder separators
        vocals = self.autoencoder(mix_input, 15, name='vocals')
        bass = self.autoencoder(mix_input, 15, name='bass')
        drums = self.autoencoder(mix_input, 3, name='drums')
        other = self.autoencoder(mix_input, 7, name='other')

        self.model = keras.Model(inputs=[mix_input], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def save_weights(self, path):
        raise NotImplementedError

    def load_weights(self, path):
        raise NotImplementedError

    def save_model_plot(self, file_name='dae_separator.png'):
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
