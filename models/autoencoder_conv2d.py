import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class AutoencoderConv2d:
    def __init__(self, freq_bins, time_frames, kernel_size=(5, 5)):
        self.bins = freq_bins
        self.frames = time_frames
        self.summary = dict()
        self.model = None
        self.kernel_size = kernel_size

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

    def get_model(self, name='autoencoder_spectrogram'):
        """FCN design, autoencoder with concat skip connection"""
        mix_input = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_input = tf.expand_dims(mix_input, axis=3)

        # encoder
        # 1st conv + downsampling + batch normalization
        conv1 = keras.layers.Conv2D(32, self.kernel_size, padding='same', activation='relu')(reshaped_input)
        downsample1 = keras.layers.Conv2D(32, self.kernel_size, strides=(2, 2), padding='same',
                                          activation='relu', use_bias=False)(conv1)
        bn1 = keras.layers.BatchNormalization()(downsample1)

        # 2nd conv + downsampling + batch normalization
        conv2 = keras.layers.Conv2D(64, self.kernel_size, padding='same', activation='relu')(bn1)
        downsample2 = keras.layers.Conv2D(64, self.kernel_size, strides=(2, 2), padding='same',
                                          activation='relu', use_bias=False)(conv2)
        bn2 = keras.layers.BatchNormalization()(downsample2)

        # 3rd conv + batch normalization
        conv3 = keras.layers.Conv2D(128, self.kernel_size, padding='same', activation='relu', use_bias=False)(bn2)
        bn3 = keras.layers.BatchNormalization()(conv3)

        # decoder
        # 4th conv + upsampling + batch normalization
        upsample1 = keras.layers.UpSampling2D((2, 2))(bn3)
        up1_fixed_shape = keras.layers.Conv2D(64, (2, 1), activation='relu', use_bias=False)(upsample1)
        conv4 = keras.layers.Conv2D(64, self.kernel_size, padding='same', activation='relu')(
            keras.layers.concatenate([conv2, up1_fixed_shape]))
        bn4 = keras.layers.BatchNormalization()(conv4)

        # 5th conv + upsampling + batch normalization
        upsample2 = keras.layers.UpSampling2D((2, 2))(bn4)
        up2_fixed_shape = keras.layers.Conv2D(32, (2, 2), activation='relu', use_bias=False)(upsample2)
        conv5 = keras.layers.Conv2D(32, self.kernel_size, padding='same', activation='relu')(
            keras.layers.concatenate([conv1, up2_fixed_shape]))
        bn5 = keras.layers.BatchNormalization()(conv5)

        # output layers
        x = keras.layers.Conv2D(32, self.kernel_size, padding='same', activation='relu')(bn5)
        x = keras.layers.Conv2D(16, self.kernel_size, padding='same', activation='relu')(x)
        output = keras.layers.Conv2D(4, self.kernel_size, padding='same', activation='relu')(x)

        # uniformly split channels into 4
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

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def save_model_plot(self, file_name='autoencoder_conv2d_separator.png'):
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
