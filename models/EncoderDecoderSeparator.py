import tensorflow as tf
from tensorflow import keras
from layers.encoder_decoder_layer import Encoder, Decoder


class Separator(keras.Model):
    """
    Encoder: compression information
    Decoder: used as separator for STEM
    """

    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='Decoder_separator',
                 **kwargs):
        super(Separator, self).__init__(name=name, **kwargs)
        self.freq_bins = frequency_bins
        self.time_frames = time_frames

        # encoder to compress input dimension
        self.encoder = Encoder(frequency_bins, time_frames)
        # decoder as separator for STEM
        self.vocals_sep = Decoder(frequency_bins, time_frames)
        self.bass_sep = Decoder(frequency_bins, time_frames)
        self.drums_sep = Decoder(frequency_bins, time_frames)
        self.other_sep = Decoder(frequency_bins, time_frames)

    @tf.function
    def call(self, inputs):
        latent_vec = self.encoder(inputs)
        vocals = self.vocals_sep(latent_vec)
        bass = self.bass_sep(latent_vec)
        drums = self.drums_sep(latent_vec)
        other = self.other_sep(latent_vec)
        return tf.stack([vocals, bass, drums, other], axis=1)
