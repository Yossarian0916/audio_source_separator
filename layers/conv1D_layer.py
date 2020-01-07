import tensorflow as tf
from tensorflow.keras import layers, Model


class Conv1DTranspose(layers.Layer):
    """1d transposed convolution layer"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 name=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(name=name, **kwargs)
        self.expand_dim = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))
        self.conv2dtranspose = layers.Conv2DTranspose(filters,
                                                      (kernel_size, 1),
                                                      (strides, 1),
                                                      padding='same')
        self.squeeze_dim = layers.Lambda(lambda x: tf.squeeze(x, axis=2))
        self.activation = layers.LeakyReLU(alpha=0.01)

    @tf.function
    def call(self, inputs):
        x = self.expand_dim(inputs)
        x = self.conv2dtranspose(x)
        x = self.squeeze_dim(x)
        x = self.activation(x)
        return x


class UpConv1D(layers.Layer):
    """deconvolution layer = upsampling + conv1D"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 name=None,
                 **kwargs):
        super(UpConv1D, self).__init__(name=name, **kwargs)
        self.upsampling = layers.UpSampling1D(size=2)
        self.conv1d = layers.Conv1D(
            filters, kernel_size, strides, padding='same')
        self.activation = layers.LekayReLU(alpha=0.01)
    
    @tf.function
    def call(self, inputs):
        x = self.upsampling(inputs)
        x = self.conv1d(x)
        x = self.activation(x)
        return x
