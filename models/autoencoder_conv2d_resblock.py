import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class AutoenocderConv2dResblock:
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
                       kernel_regularizer=None):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.ReLU()(x)
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
                   kernel_regularizer=None):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        if keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(input_tensor)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = keras.layers.ReLU()(x)
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

    def get_model(self, data_format='channels_last', name='autoencoder_conv2d_resblock'):
        """autoencoder structure, consists of resnet identity blocks"""
        # input tensor shape: (batch, height, width, channels)
        tf.keras.backend.set_image_data_format(data_format)
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_input = tf.expand_dims(mix_input, axis=-1)
        conv0 = keras.layers.Conv2D(8, self.kernel_size, padding='same',
                                    activation='relu', use_bias=False)(reshaped_input)
        bn0 = keras.layers.BatchNormalization()(conv0)

        filters_set = [[16 << i, 16 << i, 32 << i] for i in range(4)]
        # downsampling
        # Conv1 + residual identity block + maxpooling
        conv1 = self.conv_block(bn0, filters_set[0], self.kernel_size)
        res_block1 = self.identity_block(conv1, filters_set[0], self.kernel_size)
        downsample1 = keras.layers.MaxPool2D((2, 2), padding='same', data_format=data_format)(res_block1)
        # Conv2 + residual identity block + maxpooling
        conv2 = self.conv_block(downsample1, filters_set[1], self.kernel_size)
        res_block2 = self.identity_block(conv2, filters_set[1], self.kernel_size)
        downsample2 = keras.layers.MaxPool2D((2, 2), padding='same', data_format=data_format)(res_block2)
        # Conv3 + residual identity block + maxpooling
        conv3 = self.conv_block(downsample2, filters_set[2], self.kernel_size)
        res_block3 = self.identity_block(conv3, filters_set[2], self.kernel_size)
        downsample3 = keras.layers.MaxPool2D((2, 2), padding='same', data_format=data_format)(res_block3)

        # latent tensor, compressed features
        # Conv4 + residual identity block
        conv4 = self.conv_block(downsample3, filters_set[3], self.kernel_size)
        res_block4 = self.identity_block(conv4, filters_set[3], self.kernel_size)

        # upsampling
        # upsampling + residual identity block + Conv5
        upsample1 = keras.layers.UpSampling2D((2, 2), data_format=data_format)(res_block4)
        conv5 = self.conv_block(upsample1, filters_set[2], self.kernel_size)
        res_block5 = self.identity_block(self.crop(conv5, res_block3.get_shape().as_list()),
                                         filters_set[2], self.kernel_size)
        # upsampling + residual identity block + Conv6
        upsample2 = keras.layers.UpSampling2D((2, 2), data_format=data_format)(res_block5)
        conv6 = self.conv_block(upsample2, filters_set[1], self.kernel_size)
        res_block6 = self.identity_block(self.crop(conv6, res_block2.get_shape().as_list()),
                                         filters_set[1], self.kernel_size)
        # upsampling + residual identity block + Conv7
        upsample3 = keras.layers.UpSampling2D((2, 2), data_format=data_format)(res_block6)
        conv7 = self.conv_block(upsample3, filters_set[0], self.kernel_size)
        res_block7 = self.identity_block(self.crop(conv7, res_block1.get_shape().as_list()),
                                         filters_set[0], self.kernel_size)

        # output layers
        conv8 = self.conv_block(res_block7, [16, 16, 8], self.kernel_size)
        res_block8 = self.identity_block(conv8, [16, 16, 8], self.kernel_size)
        conv9 = self.conv_block(res_block7, [8, 8, 4], self.kernel_size)
        res_block9 = self.identity_block(conv8, [8, 8, 4], self.kernel_size)

        # uniformly split channels into 4
        # Lambda layer needs to assign name to the layer
        # so that during training, the outputs of the model will find
        # corresponding train data generated by pre-built tfrecord dataset
        vocals = keras.layers.Lambda(lambda x: x[:, :, :, 0], name='vocals')(res_block9)
        bass = keras.layers.Lambda(lambda x: x[:, :, :, 1], name='bass')(res_block9)
        drums = keras.layers.Lambda(lambda x: x[:, :, :, 2], name='drums')(res_block9)
        other = keras.layers.Lambda(lambda x: x[:, :, :, 3], name='other')(res_block9)

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
        return tf.concat([x1, x2], axis=-1)

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

    def save_model_plot(self, file_name='autoencoder_conv2d_resblock_separator.png'):
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
