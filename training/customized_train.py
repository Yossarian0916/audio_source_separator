import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
import sys
from utils.helper import get_data_dir_filenames
from utils.dataset import tfrecord2dataset


def load_dsd100_dataset(batch_size):
    # data dir path
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root, 'data')
    # parse tfrecords
    dsd100_train_tfrecords = get_data_dir_filenames('dsd100_train_tfrecords'+'/*')
    dsd100_test_tfrecords = get_data_dir_filenames('dsd100_test_tfrecords'+'/*')
    # training dataset
    train_tfrecords = dsd100_train_tfrecords + \
        dsd100_test_tfrecords[:len(dsd100_test_tfrecords)//2]
    train_dataset = tfrecord2dataset(train_tfrecords, batch_size)
    # validation dataset
    valid_tfrecords = dsd100_test_tfrecords[len(dsd100_test_tfrecords)//2:]
    valid_dataset = tfrecord2dataset(valid_tfrecords, batch_size)

    train_data_size = len(train_tfrecords)
    valid_data_size = len(valid_tfrecords)
    return (train_dataset, valid_dataset), (train_data_size, valid_data_size)


def save_model(model, file_name):
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    saved_model_dir = os.path.join(root, 'saved_model')
    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)
    date_time = datetime.now().strftime("%Y%m%d_%H%M")
    saved_model_path = os.path.join(
        saved_model_dir, file_name+"?time={}.h5".format(date_time))
    tf.saved_model.save(model, saved_model_path)


def train(model,
          dataset,
          epochs,
          optimizer,
          loss_fn,
          num_samples,
          batch_size,
          save_dir=None,
          save_file_name=None,):
    # collections for saving trainin results
    step_loss = list()
    epoch_loss = list()
    train_results = {'step_loss': step_loss, 'epoch_loss': epoch_loss}

    step_per_epoch = num_samples // batch_size
    # iterate over epochs
    for epoch in range(epochs):
        print('START of Epoch %d' % (epoch,))

        # iterate over batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(dataset):

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
            if step % (step_per_epoch // 10) == 0:
                print('\tTraining loss (for one batch) at step %s: %s' %
                      (step, float(loss_value)))
                print('\tSeen so far: %s samples' % ((step + 1) * batch_size))
            if step > step_per_epoch:
                break

        # save loss value of one epoch
        epoch_loss.append(
            sum(step_loss[step_per_epoch*epoch:]) / step_per_epoch)

    # save model
    if (save_dir is not None) and (save_file_name is not None):
        save_model(model, save_dir, save_file_name)

    return train_results
