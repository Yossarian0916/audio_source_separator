import tensorflow as tf
from tensorflow import keras


def identity_block(self,
                   input_tensor,
                   filters,
                   kernel_size,
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(0.01)):
    """residual block built with identity skip connection"""
    filter1, filter2, filter3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # kernel: (1, 1) layer
    x = keras.layers.Conv2D(filter1, (1, 1), padding='same', 'use_bias=False,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = keras.layers.LeakyReLU(0.01)(x)
    # kernel: kernel_size layer
    x = keras.layers.Conv2D(filter2, kernel_size, padding='same', use_bias=False,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = keras.layers.LeakyReLU(0.01)(x)
    # kernel: (1, 1) layer
    x = keras.layers.Conv2D(filter3, (1, 1), padding='same', use_bias=False,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis)(x)

    x_plus_skip_conn = keras.layers.Add()([x, input_tensor])
    output = keras.layers.LeakyReLU(0.01)(x_plus_skip_conn)
    return output
