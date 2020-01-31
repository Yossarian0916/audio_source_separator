import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class AutoencoderConv2d:
    def __init__(self, freq_bins, time_frames):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None
        self.kernel_size = (31, 5)

    def identity_block(self,
                       input_tensor,
                       filters,
                       kernel_size,
                       kernel_initializer='he_normal',
                       kernel_regularizer=keras.regularizers.l2(0.01)):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_first':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.Conv2D(filter1, (1, 1), use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=bn_axis)
        x = keras.layers.LeakyReLU(0.01)(x)
        # kernel: kernel_size layer
        x = keras.layers.Conv2D(filter2, kernel_size, use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=bn_axis)
        x = keras.layers.LeakyReLU(0.01)(x)
        # kernel: (1, 1) layer
        x = keras.layers.Conv2D(filter3, (1, 1), use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=bn_axis)

        x_plus_skip_conn = layers.Add([x, input_tensor])
        output = keras.layers.LeakyReLU(0.01)(x_plus_skip_conn)
        return output

    def downsample_block(self,
                         input_tensor,
                         filter,
                         kernel_size,
                         downsample_scale,
                         kernel_initializer='he_normal',
                         kernel_regularizer=keras.regularizers.l2(0.01)):
        freq_scale, frame_scale = downsample_scale
        if keras.backend.image_data_format() == 'channels_first':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.Conv2D(filter, kernel_size,
                                strides=(freq_scale, frame_scale),
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization(axis=bn_axis)
        x_plus_skip_conn = layers.Add([x, input_tensor])
        output = keras.layers.LeakyReLU(0.01)(x_plus_skip_conn)
        return output

    def get_model(self, name='autoencoder_spectrogram'):
        """FCN design, autoencoder with concat skip connection"""
        # input tensor shape: (batch, height, width, channels)
        tf.keras.backend.set_image_data_format(data_format)
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_input = tf.expand_dims(mix_input, axis=3)

        # downsampling
        res_block1 = self.identity_block(reshaped_input, 32, self.kernel_size)
        downsample1 = self.downsample_block(res_block1, 32, self.kernel_size, (2, 2))

        res_block2 = self.identity_block(downsample1, 64, self.kernel_size)
        downsample2 = self.downsample_block(res_block2, 64, self.kernel_size, (2, 2))

        # latent tensor, compressed features
        res_block3 = self.identity_block(downsample2, 128, self.kernel_size)

        # upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(res_block3)
        res_block4 = self.identity_block(self.crop_and_concat(upsample1, res_block2), 64, self.kernel_size)

        upsample2 = keras.layers.UpSampling2D((2, 2))(res_block4)
        res_block5 = self.identity_block(self.crop_and_concat(upsample2, res_block1), 32, self.kernel_size)

        # output layers
        res_block6 = self.identity_block(res_block5, 16, self.kernel_size)
        res_block7 = self.identity_block(res_block6, 8, self.kernel_size)
        output = self.identity_block(res_block7, 4, self.kernel_size)

        # uniformly split channels into 4
        vocals_input, bass_input, drums_input, other_input = tf.split(output, 4, axis=3)
        # Lambda layer does nothing here, just to add name to the layer
        # so that during training, the outputs of the model will find
        # corresponding train data generated by pre-built tfrecord dataset
        vocals = keras.layers.Lambda(lambda x: x, name='vocals')(vocals_input)
        bass = keras.layers.Lambda(lambda x: x, name='bass')(bass_input)
        drums = keras.layers.Lambda(lambda x: x, name='drums')(drums_input)
        other = keras.layers.Lambda(lambda x: x, name='other')(other_input)

        self.model = keras.Model(inputs=[mix_input], outputs=[vocals, bass, drums, other], name=name)
        return self.model

    def crop_and_concat(self, x1, x2):
        """
        crop tensor x1 to match x2, x2 shape is the target shape,
        then concatenate them along feature dimension
        """
        if x2 is None:
            return x1
        x1 = self.crop(x1, x2.get_shape().as_list())
        return tf.concat([x1, x2], axis=3)

    def crop(self, tensor, target_shape):
        """
        crop tensor to match target_shape,
        remove the diff/2 items at the start and at the end,
        keep only the central part of the vector
        """
        # the tensor flow in model is of shape (batch, freq_bins, time_frames, channels)
        shape = tensor.get_shape().as_list()
        diff_row = shape[1] - target_shape[1]
        diff_col = shape[2] - target_shape[2]
        assert diff_row >= 0 and diff_col >= 0  # Only positive difference allowed
        if diff_row == 0 and diff_col == 0:
            return tensor
        # calculate new cropped row index
        row_crop_start = diff_row // 2
        row_crop_end = diff_row - row_crop_start
        # calculate new cropped column index
        col_crop_start = diff_col // 2
        col_crop_end = diff_row - col_crop_start

        return tensor[:, row_crop_start:-row_crop_end, col_crop_start:-col_crop_end, :]

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
