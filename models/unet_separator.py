import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class UnetSeparator:
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

    def get_model(self, name='unet_separator'):
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # input
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')

        # downsampling
        conv1 = self.conv1d_bn(mix_input, self.frames, 15)
        maxpool1 = keras.layers.MaxPool1D(2, padding='same')(conv1)
        conv2 = self.conv1d_bn(maxpool1, self.frames, 15)
        maxpool2 = keras.layers.MaxPool1D(2, padding='same')(conv2)
        conv3 = self.conv1d_bn(maxpool2, self.frames, 15)
        maxpool3 = keras.layers.MaxPool1D(2, padding='same')(conv3)
        conv4 = self.conv1d_bn(maxpool3, self.frames, 15)

        # upsampling
        conv4_upsampling = keras.layers.UpSampling1D(2)(conv4)
        conv5_input = self.crop_and_concat(conv4_upsampling, conv3)
        conv5 = self.conv1d_bn(conv5_input, self.frames, 15, padding='same')

        conv5_upsampling = keras.layers.UpSampling1D(2)(conv5)
        conv6_input = self.crop_and_concat(conv5_upsampling, conv2)
        conv6 = self.conv1d_bn(conv6_input, self.frames, 15, padding='same')
        conv6_upsampling = keras.layers.UpSampling1D(2)(conv6)

        conv7_input = self.crop_and_concat(conv6_upsampling, conv1)
        conv7 = self.conv1d_bn(conv7_input, self.frames, 15, padding='same')

        # output layer
        output = keras.layers.Add()([conv7, mix_input])
        output = self.tile(output, (1, 4))

        # uniformly split tensor along time_frames axis into 4 inputs
        vocals_input, bass_input, drums_input, other_input = tf.split(output, 4, axis=2)
        # last conv1d layer separation
        vocals = self.conv1d_bn(vocals_input, self.frames, 15, name='vocals')
        bass = self.conv1d_bn(bass_input, self.frames, 15, name='bass')
        drums = self.conv1d_bn(drums_input, self.frames, 3, name='drums')
        other = self.conv1d_bn(other_input, self.frames, 7, name='other')

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
