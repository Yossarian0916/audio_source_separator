"""parameters for calculation STFT and hyperparameters for model"""


STFT_CONFIG = {
    "N_FFT": 2048,
    "HOP_LEN": 512,
    "WIN_LEN": 2048,
    "SR": 44100,
    "CLIP_LEN": 2.0
}

STEM_FEAT = ['mix', 'vocals', 'bass', 'drums', 'other']
