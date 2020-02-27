import os
import numpy as np
from tensorflow import keras
from utils import module_path


class SeparatorModel:
    def __init__(self, freq_bins, time_frames, kernel_size, name):
        self.model = None
        self.bins = freq_bins
        self.frames = time_frames
        self.kernel_size = kernel_size
        self.summary = dict()
        self.name = name

    def get_model(self):
        pass

    def get_name(self):
        return self.name

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_model_plot(self):
        filename = self.name + '.png'
        if self.model is not None:
            root_dir = module_path.get_root_path()
            images_dir = os.path.join(root_dir, 'images')
            if not os.path.exists(images_dir):
                os.mkdir(images_dir)
            file_path = os.path.join(images_dir, filename)
            keras.utils.plot_model(self.model, file_path)
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

    def model_summary(self):
        if self.model is not None:
            trainable_count = np.sum([keras.backend.count_params(w)
                                      for w in self.model.trainable_weights])
            non_trainable_count = np.sum([keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            self.summary = {
                'total_parameters': trainable_count + non_trainable_count,
                'trainable_parameters': trainable_count,
                'non_trainable_parameters': non_trainable_count}
        else:
            raise ValueError("no model has been built yet! call get_model() first!")

    def __str__(self):
        return self.summary

    __repr__ = __str__
