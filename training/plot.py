import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from datetime import datetime
import os
import sys


def plot_curve(results, size=(16, 9)):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=size)
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].plot(results)
    plt.show()


def plot_model(model, name):
    keras.utils.plot_model(model, name, show_shapes=True)
