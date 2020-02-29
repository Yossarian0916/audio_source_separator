import tensorflow as tf
from tensorflow import keras
from models.separator_model import SeparatorModel
import models.util as util


class ConvEncoderDenoisingDecoder(SeparatorModel):
    def __init__(self, freq_bins, time_frames,
                 kernel_size=(3, 3),
                 kernel_initializer='he_normal',
                 regularization=keras.regularizers.l1(0.001),
                 name='conv_encoder_denoising_decoder'):
        super(ConvEncoderDenoisingDecoder, self).__init__(freq_bins, time_frames, kernel_size, name)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularization = regularization

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   padding='same',
                   use_bias=True):
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                activation='relu',
                                use_bias=use_bias,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(input_tensor)
        x = keras.layers.BatchNormalization()(x, training=True)
        return x

    def encoder(self, input_tensor):
        x = self.conv_block(input_tensor, 4, self.kernel_size)
        # 1st downsampling
        conv1 = self.conv_block(x, 8, self.kernel_size)
        downsample1 = self.conv_block(conv1, 8, self.kernel_size, strides=(2, 2))
        # 2nd downsampling
        conv2 = self.conv_block(downsample1, 16, self.kernel_size)
        downsample2 = self.conv_block(conv2, 16, self.kernel_size, strides=(2, 2))
        # 3rd downsampling
        conv3 = self.conv_block(downsample1, 32, self.kernel_size)
        downsample3 = self.conv_block(conv3, 32, self.kernel_size, strides=(2, 2))
        # intermediate low dimensional features
        x = self.conv_block(downsample3, 32, self.kernel_size)
        x = self.conv_block(x, 64, self.kernel_size)
        latent_space = self.conv_block(x, 32, self.kernel_size)
        return latent_space, conv3, conv2, conv1

    def decoder(self, input_tensors, name=None):
        latent_space, conv3, conv2, conv1 = input_tensors
        # 1st upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(latent_space)
        x = self.conv_block(util.crop_and_concat(upsample1, conv3), 16, self.kernel_size)
        # 2nd upsampling
        upsample2 = keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_block(util.crop_and_concat(upsample2, conv2), 8, self.kernel_size)
        # 3rd upsampling
        upsample3 = keras.layers.UpSampling2D((2, 2))(x)
        x = self.conv_block(util.crop_and_concat(upsample3, conv1), 4, self.kernel_size)
        output = keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='relu',
                                     kernel_initializer='he_normal', name=name)(x)
        return output

    def get_model(self):
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        (latent, conv3, conv2, conv1) = self.encoder(reshaped_spectrogram)
        vocals = self.decoder((latent, conv3, conv2, conv1), name='vocals')
        bass = self.decoder((latent, conv3, conv2, conv1), name='bass')
        drums = self.decoder((latent, conv3, conv2, conv1), name='drums')
        other = self.decoder((latent, conv3, conv2, conv1), name='other')

        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=self.name)
        return self.model
