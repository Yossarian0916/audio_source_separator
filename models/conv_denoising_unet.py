import tensorflow as tf
from tensorflow import keras
from models.model_template import SeparatorModel
import models.util as util


class ConvDenoisingUnet(SeparatorModel):
    def __init__(self, freq_bins, time_frames, kernel_size=(3, 3), name='conv_denoising_unet'):
        super(ConvDenoisingUnet, self).__init__(freq_bins, time_frames, kernel_size, name)

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(0.001)):
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(input_tensor)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x, training=True)
        return x

    def unet(self, name='unet'):
        """crop-and-concat features as skip connection, fully convolutional structure"""
        spectrogram = keras.Input(shape=(self.bins, self.frames, 1))

        conv0 = self.conv_block(spectrogram, 4, (1, 1))
        # encoder
        # 1st downsampling
        conv1 = self.conv_block(conv0, 8, self.kernel_size)
        downsample1 = self.conv_block(conv1, 8, self.kernel_size, strides=(2, 2))

        # 2nd downsampling
        conv2 = self.conv_block(downsample1, 16, self.kernel_size)
        downsample2 = self.conv_block(conv2, 16, self.kernel_size, strides=(2, 2))

        # intermediate low dimensional features
        conv3 = self.conv_block(downsample2, 32, self.kernel_size)
        conv4 = self.conv_block(conv3, 32, self.kernel_size)

        # decoder
        # 1st upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(conv4)
        conv5 = self.conv_block(util.crop_and_concat(upsample1, conv2), 16, self.kernel_size)

        # 2nd upsampling
        upsample2 = keras.layers.UpSampling2D((2, 2))(conv5)
        conv6 = self.conv_block(util.crop_and_concat(upsample2, conv1), 8, self.kernel_size)

        # output layers
        output = self.conv_block(conv6, 1, kernel_size=(1, 1))
        return keras.Model(inputs=[spectrogram], outputs=[output], name=name)

    def get_model(self):
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        vocals = self.unet(name='vocals')(reshaped_spectrogram)
        bass = self.unet(name='bass')(reshaped_spectrogram)
        drums = self.unet(name='drums')(reshaped_spectrogram)
        # other = keras.layers.Subtract(name='other')([reshaped_spectrogram, (vocals + bass + drums)])
        other = self.unet(name='other')(reshaped_spectrogram)

        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=self.name)
        return self.model
