import tensorflow as tf
from tensorflow import keras
from layers.conv1D_layer import Conv1DTranspose
from layers.encoder_decoder_layer import Encoder, Decoder


class Autoencoder(keras.Model):
    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='autoencoder_skip_conn',
                 **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(frequency_bins, time_frames)
        self.decoder = Decoder(frequency_bins, time_frames)

    @tf.function
    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed


class AutoencoderExpandDim(keras.Model):
    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='autoencoder_expand_output_dim',
                 **kwargs):
        super(AutoencoderExpandDim, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(frequency_bins, time_frames)
        self.decoder = Decoder(frequency_bins, time_frames)
        self.expand_dim = keras.Sequential(
            layers=[
                keras.layers.Reshape((time_frames, frequency_bins)),
                Conv1DTranspose(filters=frequency_bins*4, kernel_size=3),
                keras.layers.Reshape((frequency_bins*4, time_frames))
            ]
        )

    @tf.function
    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        output = self.expand_dim(reconstructed)
        return output
