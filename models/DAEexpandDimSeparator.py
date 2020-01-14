import tensorflow as tf
from tensorflow import keras
from models.autoencoder import Autoencoder, AutoencoderExpandDim


class Separator(keras.Model):
    """
    Reconstruction: denoising autoencoder, output is 4 times expanded original frequency dimension
    Separator: for each STEM is a separate denoising autoencoder
    """

    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='Denoising_autoencoder_separator',
                 **kwargs):
        super(Separator, self).__init__(name=name, **kwargs)
        self.freq_bins = frequency_bins
        self.time_frames = time_frames

        # autoencoder for reconstruction, output with expanded dimension (4*frequency_bins)
        self.reconstruction = AutoencoderExpandDim(frequency_bins, time_frames)
        # denoising autoencoder separator for STEM
        self.vocals_sep = Autoencoder(frequency_bins, time_frames)
        self.bass_sep = Autoencoder(frequency_bins, time_frames)
        self.drums_sep = Autoencoder(frequency_bins, time_frames)
        self.other_sep = Autoencoder(frequency_bins, time_frames)

    @tf.function
    def call(self, inputs):
        recon_expand_dim = self.reconstruction(inputs)
        vocals = self.vocals_sep(recon_expand_dim[:self.freq_bins])
        bass = self.bass_sep(recon_expand_dim[self.freq_bins:self.freq_bins*2])
        drums = self.drums_sep(
            recon_expand_dim[self.freq_bins*2:self.freq_bins*3])
        other = self.other_sep(recon_expand_dim[self.freq_bins*3:])
        return tf.stack([vocals, bass, drums, other], axis=1)
