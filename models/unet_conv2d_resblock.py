import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class UnetConv2dResblock:
    def __init__(self, freq_bins, time_frames, kernel_size):
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
                       kernel_regularizer=keras.regularizers.l2(0.01)):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter1, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter2, kernel_size,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter3, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)
        output = keras.layers.Add()([x, input_tensor])
        return output

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(0.01)):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter1, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter2, kernel_size,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.LeakyReLU(0.01)(x)
        x = keras.layers.Conv2D(filter3, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)
        # skip connection
        shortcut = keras.layers.Conv2D(filter3, (1, 1),
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer=kernel_initializer,
                                       kernel_regularizer=kernel_regularizer)(input_tensor)
        shortcut = keras.layers.BatchNormalization(axis=bn_axis)(shortcut)
        output = keras.layers.Add()([x, shortcut])
        return output

    def get_model(self, data_format='channels_last', name='unet_conv2d_resblock'):
        """UNet structure based, consists of resnet identity blocks"""
        # input tensor shape: (batch, height, width, channels)
        tf.keras.backend.set_image_data_format(data_format)
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_input = tf.expand_dims(mix_input, axis=3)

        # downsampling
        conv1 = self.conv_block(reshaped_input, [16, 16, 32], self.kernel_size)
        res_block1 = self.identity_block(conv1, [16, 16, 32], self.kernel_size)
        downsample1 = keras.layers.MaxPool2D((2, 2), padding='same', data_format=data_format)(res_block1)
        conv2 = self.conv_block(downsample1, [32, 32, 64], self.kernel_size)
        res_block2 = self.identity_block(conv2, [32, 32, 64], self.kernel_size)
        downsample2 = keras.layers.MaxPool2D((2, 2), padding='same', data_format=data_format)(res_block2)

        # latent tensor, compressed features
        conv3 = self.conv_block(downsample2, [64, 64, 128], self.kernel_size)
        res_block3 = self.identity_block(conv3, [64, 64, 128], self.kernel_size)

        # upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2), data_format=data_format)(res_block3)
        res_block4 = self.identity_block(self.crop(upsample1, res_block2.get_shape().as_list()),
                                         [128, 128, 64], self.kernel_size)

        upsample2 = keras.layers.UpSampling2D((2, 2), data_format=data_format)(res_block4)
        res_block5 = self.identity_block(self.crop(upsample2, res_block1.get_shape().as_list()),
                                         [64, 64, 32], self.kernel_size)

        # output layers
        conv4 = self.conv_block(res_block5, [32, 32, 16], self.kernel_size)
        res_block6 = self.identity_block(conv4, [32, 32, 16], self.kernel_size)
        conv5 = self.conv_block(res_block6, [16, 16, 8], self.kernel_size)
        res_block7 = self.identity_block(conv5, [16, 16, 8], self.kernel_size)
        conv6 = self.conv_block(res_block7, [8, 8, 4], self.kernel_size)
        output = self.identity_block(conv6, [8, 8, 4], self.kernel_size)

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
        diff_channel = shape[3] - target_shape[3]
        if diff_row < 0 or diff_col < 0 or diff_channel < 0:
            raise ValueError('input tensor cannot be shaped as target_shape')
        row_slice = slice(0, None)
        col_slice = slice(0, None)
        channel_slice = slice(0, None)
        if diff_row == 0 and diff_col == 0 and diff_channel == 0:
            return tensor
        if diff_row != 0:
            # calculate new cropped row index
            row_crop_start = diff_row // 2
            row_crop_end = diff_row - row_crop_start
            row_slice = slice(row_crop_start, -row_crop_end)
        if diff_col != 0:
            # calculate new cropped column index
            col_crop_start = diff_col // 2
            col_crop_end = diff_col - col_crop_start
            col_slice = slice(col_crop_start, -col_crop_end)
        if diff_channel != 0:
            # calculate new cropped channel axis index
            channel_crop_start = diff_channel // 2
            channel_crop_end = diff_channel - channel_crop_start
            channel_slice = slice(channel_crop_start, -channel_crop_end)
        return tensor[:, row_slice, col_slice, channel_slice]

    def save_weights(self, path):
        raise NotImplementedError

    def load_weights(self, path):
        raise NotImplementedError

    def save_model_plot(self, file_name='unet_resblock_spectro_separator.png'):
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
