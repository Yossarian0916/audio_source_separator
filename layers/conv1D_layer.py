import tensorflow as tf
from tensorflow import keras


class Conv1DTranspose(keras.layers.Layer):
    """1d transposed convolution"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 name=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(name=name, **kwargs)
        self.expand_dim = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=2))
        self.conv2dtranspose = keras.layers.Conv2DTranspose(filters,
                                                            (kernel_size, 1),
                                                            (strides, 1),
                                                            padding='same')
        self.squeeze_dim = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

    @tf.function
    def call(self, inputs):
        x = self.expand_dim(inputs)
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
        self.conv1d = keras.layers.Conv1D(
            filters, kernel_size, strides, padding='same')

    @tf.function
    def call(self, inputs):
        x = self.upsampling(inputs)
        x = self.conv1d(x)
        return x
