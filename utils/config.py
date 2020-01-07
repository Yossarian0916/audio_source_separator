"""parameters for calculation STFT and hyperparameters for model"""


STFT_CONFIG = {
    "N_FFT": 4096,
    "HOP_LEN": 1024,
    "WIN_LEN": 4096,
    "FREQ_BINS": 2049,
    "TIME_FRAMES": 87,
    "SR": 44100,
    "CLIP_LEN": 2.0
}

DATASET = {
    "BATCH_SIZE": 256
}

TRAINING = {
    "EPOCHS": 100
}

STEM_FEAT = ['mix', 'vocals', 'bass', 'drums', 'other']