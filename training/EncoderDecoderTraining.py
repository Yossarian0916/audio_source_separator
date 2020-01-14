import tensorflow as tf
from tensorflow import keras

import os

import models.EncoderDecoderSeparator
from utils.helper import get_filenames
from utils.dataset import tfrecord2dataset


# stft transformation parameter config
N_FFT = 4096
HOP_LEN = 1024
WIN_LEN = 4096
FREQ_BINS = 2049
TIME_FRAMES = 87
SR = 44100
DURATION = 2.0
# dataset config
BATCH_SIZE = 64

# dataset preparation
current_path = os.path.abspath(__file__)
training_dir = os.path.abspath(current_path)
root = os.path.dirname(training_dir)
data_dir = os.path.join(root, 'data')
# tfrecords
dsd100_train_dir = os.path.join(data_dir, 'dsd100_train_tfrecords')
dsd100_train_tfrecords = get_filenames(dsd100_train_dir+'/*')
dsd100_test_dir = os.path.join(data_dir, 'dsd100_test_tfrecords')
dsd100_test_tfrecords = get_filenames(dsd100_test_dir+'/*')
# training dataset
train_tfrecords = dsd100_train_tfrecords + dsd100_test_tfrecords
train_dataset = tfrecord2dataset(train_tfrecords, batch_size=BATCH_SIZE)
# validation dataset
valid_tfrecords = dsd100_test_tfrecords[:len(dsd100_test_tfrecords)//2]
valid_dataset = tfrecord2dataset(valid_tfrecords, batch_size=BATCH_SIZE)
# test dataset
test_tfrecords = dsd100_test_tfrecords[len(dsd100_test_tfrecords)//2:]
test_dataset = tfrecord2dataset(test_tfrecords, batch_size=BATCH_SIZE)

# training config
TRAIN_DATA_SIZE = len(train_tfrecords)
VAL_DATA_SIZE = len(valid_tfrecords)
TEST_DATA_SIZE = len(test_tfrecords)


def train(model,
          dataset,
          epochs,
          optimizer,
          loss_fn,
          num_samples=TRAIN_DATA_SIZE,
          batch_size=BATCH_SIZE,):
    # collections for saving trainin results
    step_loss = list()
    epoch_loss = list()
    train_results = {'step_loss': step_loss, 'epoch_loss': epoch_loss}

    step_per_epoch = num_samples // batch_size
    # iterate over epochs
    for epoch in range(epochs):
        print('START of Epoch %d' % (epoch,))

        # iterate over batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train)
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_pred, y_batch_train)
            # run one step of gradient descent to update weights
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # save loss value of one step
            step_loss.append(loss_value)
            if step % 50 == 0:
                print('\tTraining loss (for one batch) at step %s: %s' %
                      (step, float(loss_value)))
                print('\tSeen so far: %s samples' % ((step + 1) * batch_size))
            if step > step_per_epoch:
                break

        # save loss value of one epoch
        epoch_loss.append(
            sum(step_loss[step_per_epoch*epoch:]) / step_per_epoch)

    return train_results


if __name__ == '__main__':
    epochs = 100
    lr = 0.0005
    separator = models.EncoderDecoderSeparator.Separator(
        frequency_bins=FREQ_BINS, time_frames=TIME_FRAMES)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    mse_loss = tf.keras.losses.MeanSquaredError()

    results = train(separator, train_dataset, epochs,
                    optimizer=adam, loss_fn=mse_loss)
