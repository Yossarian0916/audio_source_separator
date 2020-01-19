import tensorflow as tf
from tensorflow import keras


# 1d transposed convolution
def conv1DTranspose(filters, kernel_size=3, strides=1, padding='same', activation=None, kernel_initializer='glorot_uniform'):
    tconv1d = keras.Sequential(
    [
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)),
        keras.layers.Conv2DTranspose(filters=filters,
                                     kernel_size=(kernel_size, 1),
                                     strides=(strides, 1),
                                     padding=padding,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer),
        keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))
    ])
    return tconv1d


class Conv1D_BN(keras.layers.Layer):
    """1d convolution, LeakyReLU activation, followed by batch normalization"""
    
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_normal',
                 training=False,
                 name=None,
                 **kwargs):
        super(Conv1D_BN, self).__init__(name=name, **kwargs)
        self.training = training
        self.conv1d = keras.layers.Conv1D(filters=filters, 
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          activation=None,
                                          kernel_initializer=kernel_initializer)
        self.activation = keras.layers.LeakyReLU(alpha=0.01)
        self.bn = keras.layers.BatchNormalization()
        
    def get_config(self):
        return {'training': self.training}
    
    @tf.function
    def call(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.bn(x, training=self.training)
        return x


class TConv1D(keras.layers.Layer):
    """1d transposed convolution, LeakyReLU activation"""
    
    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_normal',
                 name=None,
                 **kwargs):
        super(TConv1D, self).__init__(name=name, **kwargs)
        self.tconv1d = conv1DTranspose(filters=filters, 
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       activation=None,
                                       kernel_initializer=kernel_initializer)
        self.activation = keras.layers.LeakyReLU(alpha=0.01)
    
    def get_config(self):
        pass
    
    @tf.function
    def call(self, x):
        x = self.tconv1d(x)
        x = self.activation(x)
        return x


class Conv1DTranspose(keras.layers.Layer):
    """1d transposed convolution"""
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation=None,
                 kernel_initializer='glorot_normal',
                 name=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(name=name, **kwargs)
        self.expand_dim = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)),
        self.conv2dtranspose = keras.layers.Conv2DTranspose(filters=filters,
                                     kernel_size=(kernel_size, 1),
                                     strides=(strides, 1),
                                     padding=padding,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer),
        self.squeeze_dim = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))
        
    @tf.function
    def call(self, x):
        x = self.expand_dim(x)
        x = self.conv2dtranspose(x)
        x = self.squeeze_dim(x)
        return x


class UpConv1D(keras.layers.Layer):
    """upsampling + conv1D"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 name=None,
                 **kwargs):
        super(UpConv1D, self).__init__(name=name, **kwargs)
        self.upsampling = keras.layers.UpSampling1D(size=2)
        self.conv1d = keras.layers.Conv1D(filters, kernel_size, strides, padding='same')

    @tf.function
    def call(self, x):
        x = self.upsampling(x)
        x = self.conv1d(x)
        return x
