import tensorflow as tf
from tensorflow import keras
from layers.conv1D_layer import Conv1D_BN, TConv1D


class AutoencoderSeparator(keras.Model):
    """autoencoder separator, conv1d+BN+MaxPooling as encoder layers, tconv1d+upsampling as decoder layers"""
    
    def __init__(self,
                 freq_bins, 
                 time_frames,
                 training=False,
                 name='autoencoder_separator',
                 **kwargs):
        super(AutoencoderSeparator, self).__init__(name=name, **kwargs)
        self.freq_bins = freq_bins
        
        self.input_reshape = keras.layers.Reshape((time_frames, freq_bins))
 
        self.conv1 = Conv1D_BN(freq_bins, kernel_size=5, strides=1, training=training)
        self.conv2 = Conv1D_BN(freq_bins * 2, kernel_size=5, strides=1, training=training)
        self.conv3 = Conv1D_BN(freq_bins * 4, kernel_size=5, strides=1, training=training)
        self.maxpool = keras.layers.MaxPool1D(pool_size=2, padding='same')
        
        self.tconv4 = TConv1D(freq_bins * 2, kernel_size=5)
        self.tconv5 = TConv1D(freq_bins, kernel_size=5)
        self.upsampling = keras.layers.UpSampling1D(size=2)
        
        self.expand_dim = Conv1D_BN(freq_bins*4, kernel_size=2, padding='valid')
        self.output_reshape = keras.layers.Reshape((freq_bins*4, time_frames))

    
    @tf.function
    def call(self, x):
        x = self.input_reshape(x)
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        
        tconv4 = self.tconv4(self.upsampling(conv3))
        tconv5 = self.tconv5(self.upsampling(tconv4))
        
        tconv6 = self.expand_dim(tconv5)
        output = self.output_reshape(tconv6)
        
        vocals = output[:, :self.freq_bins, :]
        bass = output[:, self.freq_bins:self.freq_bins * 2, :]
        drums = output[:, self.freq_bins * 2:self.freq_bins * 3, :]
        other = output[:, self.freq_bins * 3:, :]
        return tf.stack([vocals, bass, drums, other], axis=1)
