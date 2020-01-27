import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def plot_curve(results, size=(16, 9)):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=size)
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].plot(results)
    plt.show()


def plot_model(model, name='model.png'):
    tf.keras.utils.plot_model(model, name, show_shapes=True)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(16, 10))
    plt.grid(True)
    plt.show()
