import tensorflow as tf
from tensorflow import keras

from layers.conv1D_layer import Conv1DTranspose


class ResBlockConv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation=None,
                 kernel_initializer='glorot_normal',
                 name=None,
                 **kwargs):
        super(ResBlockConv, self).__init__(name=name, **kwargs)
        self.conv1d = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          activation=None,
                                          kernel_initializer=kernel_initializer)
        self.activation = keras.layers.LekayReLU(alpha=0.01)

    def get_config(self):
        pass
    
    @tf.function
    def call(self, x):
        residual = x
        out = self.conv1d(out)
        out = out + residual
        return self.activation(out)


class ResBlockTConv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation=None,
                 kernel_initializer='glorot_normal',
                 name=None,
                 **kwargs):
        super(ResBlockTConv, self).__init__(name=name, **kwargs)
        self.tconv1d = Conv1DTranspose(filters=filters, 
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       activation=None,
                                       kernel_initializer=kernel_initializer)
        self.activation = self.activation = keras.layers.LekayReLU(alpha=0.01)

    def get_config(self):
        pass
    
    @tf.function
    def call(self, x):
        residual = x
        out = self.tconv1d(out)
        out = out + residual
        return self.activation(out)
