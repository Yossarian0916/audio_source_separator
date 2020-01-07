import tensorflow as tf
from tensorflow import keras

import os

import models.DAEexpandDimSeparator
from utils.helper import get_filenames, istft
from utils.dataset import tfrecord2dataset


# parameter config
N_FFT = 4096
HOP_LEN = 1024
WIN_LEN = 4096
FREQ_BINS = 2049
TIME_FRAMES = 87
SR = 44100
DURATION = 2.0

# dataset preparation
current_path = os.path.abspath(__file__)
training_dir = os.path.dirname(current_path)
root = os.path.dirname(training_dir)
data_dir = os.path.join(root, 'data')
# training data
train_data_dir = os.path.join(data_dir, 'dsd100_train_tfrecords')
train_tfrecords_zipfiles = get_filenames(train_data_dir+'/*')
train_dataset = tfrecord2dataset(train_tfrecords_zipfiles)
# test_data
test_data_dir = os.path.join(data_dir, 'dsd100_test_tfrecords')
test_tfrecords_zipfiles = get_filenames(test_data_dir+'/*')
test_dataset = tfrecord2dataset(test_tfrecords_zipfiles)


model = models.DAEexpandDimSeparator.Separator(
    frequency_bins=FREQ_BINS, time_frames=TIME_FRAMES)
# learning rate decay function
lr_fn = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0003,
    decay_steps=10000,
    decay_rate=0.8,
    staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_fn)
mse_loss_fn = keras.losses.MeanSquaredError()

train_loss_results = []
train_accuracy_results = []
epoch_loss_history = keras.metrics.Mean()

EPOCHS = 150
for epoch in range(EPOCHS):
    print('START of Epoch %d' % (epoch,))
    for step, x_batch_train, y_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train)
            loss = tf.reduce_mean(mse_loss_fn(y_pred, y_batch_train))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # add current batch loss
        epoch_loss_avg(loss)
        if step % 10 == 0:
            print('step {}: mean loss = {}'.format(
                step, epoch_loss_history.result()))

    # after each epoch
    train_loss_results.append(epoch_loss_history.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(
            epoch, epoch_loss_history.result()))
print('END')
