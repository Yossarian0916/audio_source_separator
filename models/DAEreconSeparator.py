import tensorflow as tf
from tensorflow import keras
from models.autoencoder import Autoencoder, AutoencoderExpandDim


class Separator(keras.Model):
    """
    Reconstruction: using denoising autoencoder to reconstruct the audio
    Separator: an autoencoder for each STEM
    """

    def __init__(self,
                 frequency_bins,
                 time_frames,
                 name='Denoising_autoencoder_reconstruction_separator',
                 **kwargs):
        super(Separator, self).__init__(name=name, **kwargs)
        self.freq_bins = frequency_bins
        self.time_frames = time_frames
        # denoising autoencoder reconstructed input music STFT
        self.reconstruction = Autoencoder(frequency_bins, time_frames)
        # denoising autoencoder separator for STEM
        self.vocals_sep = Autoencoder(frequency_bins, time_frames)
        self.bass_sep = Autoencoder(frequency_bins, time_frames)
        self.drums_sep = Autoencoder(frequency_bins, time_frames)
        self.other_sep = Autoencoder(frequency_bins, time_frames)

    @tf.function
    def call(self, inputs):
        reconstruction_vec = self.reconstruction(inputs)
        vocals = self.vocals_sep(reconstruction_vec)
        bass = self.bass_sep(reconstruction_vec)
        drums = self.drums_sep(reconstruction_vec)
        other = self.other_sep(reconstruction_vec)
        return tf.stack([vocals, bass, drums, other], axis=1)
