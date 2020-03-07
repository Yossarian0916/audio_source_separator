import tensorflow as tf
from tensorflow import keras
from models.separator_model import SeparatorModel
import models.util as util


class ConvDenoisingUnet(SeparatorModel):
    def __init__(self, freq_bins, time_frames,
                 kernel_size=(3, 3),
                 kernel_initializer='he_normal',
                 regularization=keras.regularizers.l1(0.001),
                 name='conv_denoising_unet'):
        super(ConvDenoisingUnet, self).__init__(freq_bins, time_frames, kernel_size, name)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularization = regularization

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(1, 1),
                   relu_neg_slope=0,
                   padding='same',
                   use_bias=False):
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularization)(input_tensor)
        x = keras.layers.ReLU(negative_slope=relu_neg_slope)(x)
        x = keras.layers.BatchNormalization()(x, training=True)
        return x

    def unet(self, name='unet'):
        """crop-and-concat features as skip connection, fully convolutional structure"""
        spectrogram = keras.Input(shape=(self.bins, self.frames, 1))

        conv0 = self.conv_block(spectrogram, 4, (1, 1), relu_neg_slope=0.2)
        # encoder
        # 1st downsampling
        conv1 = self.conv_block(conv0, 16, self.kernel_size, relu_neg_slope=0.2)
        downsample1 = self.conv_block(conv1, 16, self.kernel_size, strides=(2, 2), relu_neg_slope=0.2)

        # 2nd downsampling
        conv2 = self.conv_block(downsample1, 32, self.kernel_size, relu_neg_slope=0.2)
        downsample2 = self.conv_block(conv2, 32, self.kernel_size, strides=(2, 2), relu_neg_slope=0.2)

        # 3rd downsampling
        conv3 = self.conv_block(downsample2, 64, self.kernel_size, relu_neg_slope=0.2)
        downsample3 = self.conv_block(conv3, 64, self.kernel_size, strides=(2, 2), relu_neg_slope=0.2)

        # intermediate low dimensional features
        conv4 = self.conv_block(downsample3, 128, self.kernel_size, relu_neg_slope=0.2)
        conv5 = self.conv_block(conv4, 128, self.kernel_size)

        # decoder
        # 1st upsampling
        upsample1 = keras.layers.UpSampling2D((2, 2))(conv5)
        conv6 = self.conv_block(util.crop_and_concat(upsample1, conv3), 64, self.kernel_size)

        # 2nd upsampling
        upsample2 = keras.layers.UpSampling2D((2, 2))(conv6)
        conv7 = self.conv_block(util.crop_and_concat(upsample2, conv2), 32, self.kernel_size)

        # 3rd upsampling
        upsample3 = keras.layers.UpSampling2D((2, 2))(conv7)
        conv8 = self.conv_block(util.crop_and_concat(upsample3, conv1), 16, self.kernel_size)

        # output layers
        output = self.conv_block(conv8, 1, kernel_size=(1, 1))
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
