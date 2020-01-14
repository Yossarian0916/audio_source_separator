import tensorflow as tf
from tensorflow import keras

from layers.conv1D_layer import Conv1DTranspose


class BlockConv(keras.layers.Layer):
    def __init__(self,
                 filter_size,
                 kernel_size=3,
                 name='BlockConv',
                 **kwargs):
        super(BlockConv, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.conv1d = keras.layers.Conv1D(filters=self.filter_size,
                                          kernel_size=self.kernel_size,
                                          padding='same')
        self.activation = self.activation = keras.layers.LekayReLU(alpha=0.01)

    @tf.function
    def call(self, inputs):
        residual = inputs
        out = self.conv1d(out)
        out = out + residual
        return self.activation(out)


class BlockTConv(keras.layers.Layer):
    def __init__(self,
                 filter_size,
                 kernel_size=3,
                 name='BlockTConv',
                 **kwargs):
        super(BlockTConv, self).__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.tconv1d = Conv1DTranspose(filters=self.filter_size,
                                       kernel_size=self.kernel_size)
        self.activation = self.activation = keras.layers.LekayReLU(alpha=0.01)

    @tf.function
    def call(self, inputs):
        residual = inputs
        out = self.tconv1d(out)
        out = out + residual
        return self.activation(out)
