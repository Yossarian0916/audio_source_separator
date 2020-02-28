import tensorflow as tf
from tensorflow import keras
from models.separator_model import SeparatorModel
import models.util as util


class ConvResblockDenoisingUnet(SeparatorModel):
    def __init__(self, freq_bins, time_frames,
                 kernel_size=(3, 3),
                 kernel_initialzer='he_normal',
                 regularization=keras.regularizers.l2(0.001),
                 name='conv_resblock_denoising_unet'):
        super(ConvResblockDenoisingUnet, self).__init__(freq_bins, time_frames, kernel_size, name)
        self.kernel_initializer = kernel_initialzer
        self.kernel_regularization = regularization

    def identity_block(self,
                       input_tensor,
                       filters,
                       kernel_size):
        """residual block built with identity skip connection"""
        filter1, filter2, filter3 = filters
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=-1)(input_tensor, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1), padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=-1)(x, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size, padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=-1)(x, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter3, (1, 1), padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)
        output = keras.layers.Add()([x, input_tensor])
        return output

    def conv_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   strides=(2, 2)):
        """
        residual block with (average pooling + convolution) skip connection, 
        changing input tensor channels,
        downsampling input tensor
        """
        filter1, filter2, filter3 = filters
        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=-1)(input_tensor, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter1, (1, 1), padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)

        # kernel: kernel_size layer
        x = keras.layers.BatchNormalization(axis=-1)(x, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter2, kernel_size, strides=strides, padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)

        # kernel: (1, 1) layer
        x = keras.layers.BatchNormalization(axis=-1)(x, training=True)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(filter3, (1, 1), padding='same',
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer)(x)
        # skip connection, change the input tensor feature channels
        if strides == (2, 2):
            shortcut = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(input_tensor)
        elif strides == (1, 1):
            shortcut = input_tensor
        shortcut = keras.layers.BatchNormalization(axis=-1)(shortcut, training=True)
        shortcut = keras.layers.LeakyReLU()(shortcut)
        shortcut = keras.layers.Conv2D(filter3, (1, 1), padding='same',
                                       use_bias=False,
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self.kernel_regularizer)(shortcut)
        output = keras.layers.Add()([x, shortcut])
        return output

    def residual_block(self, input_tensor, filters, downsample=False):
        if downsample:
            x = self.conv_block(input_tensor, filters, self.kernel_size, strides=(2, 2))
        else:
            x = self.conv_block(input_tensor, filters, self.kernel_size, strides=(1, 1))
        x = self.identity_block(x, filters, self.kernel_size)
        x = self.identity_block(x, filters, self.kernel_size)
        return x

    def resblock_unet(self, name=None):
        spectrogram = keras.Input(shape=(self.bins, self.frames, 1))
        conv0 = keras.layers.Conv2D(4, (1, 1), padding='same')(spectrogram)
        block0 = self.residual_block(conv0, [1, 1, 4])

        filters_set = [[4 << i, 4 << i, 8 << i] for i in range(4)]
        # encoder
        # residual block, 1st downsampling
        block1 = self.residual_block(block0, filters_set[0], downsample=True)
        # residual block, 2nd downsampling
        block2 = self.residual_block(block1, filters_set[1], downsample=True)
        # residual block, 3rd downsampling
        block3 = self.residual_block(block2, filters_set[2], downsample=True)

        # latent tensor, compressed low dimensional features
        latent = self.residual_block(block3, filters_set[3])
        block4 = self.residual_block(latent, filters_set[2])

        # decoder
        # 1st upsampling + residual block
        upsample1 = keras.layers.UpSampling2D((2, 2))(block4)
        block5 = self.residual_block(util.crop_and_concat(upsample1, block2), filters_set[1])
        # 2nd upsampling + residual block
        upsample2 = keras.layers.UpSampling2D((2, 2))(block5)
        block6 = self.residual_block(util.crop_and_concat(upsample2, block1), filters_set[0])
        # residual block + 3rd upsampling
        upsample3 = keras.layers.UpSampling2D((2, 2))(block6)
        block7 = self.residual_block(util.crop_and_concat(upsample3, conv0), filters_set[0])

        # output layers
        output = keras.layers.BatchNormalization(axis=-1)(block7, training=True)
        output = keras.layers.Conv2D(1, (1, 1), padding='same', activation='relu')(output)
        return keras.Model(inputs=[spectrogram], outputs=[output], name=name)

    def get_model(self):
        """autoencoder with resnet as encoder"""
        # dataset spectrogram output tensor shape: (batch, frequency_bins, time_frames)
        # add one extra channel dimension to match model required tensor shape
        spectrogram = keras.Input(shape=(self.bins, self.frames), name='mix')
        reshaped_spectrogram = tf.expand_dims(spectrogram, axis=-1)

        vocals = self.resblock_unet(name='vocals')(reshaped_spectrogram)
        bass = self.resblock_unet(name='bass')(reshaped_spectrogram)
        drums = self.resblock_unet(name='drums')(reshaped_spectrogram)
        other = self.resblock_unet(name='other')(reshaped_spectrogram)
        self.model = keras.Model(inputs=[spectrogram], outputs=[vocals, bass, drums, other], name=self.name)
        return self.model
