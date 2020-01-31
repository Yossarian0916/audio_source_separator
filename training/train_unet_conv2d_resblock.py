#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
from models.unet_conv2d_resblock import UnetConv2dResblock
from training.make_dataset import DSD100Dataset


# hyper-parameter
BATCH_SIZE = 8

# load dataset
dsd100_dataset = DSD100Dataset(batch_size=BATCH_SIZE)
train_dataset, valid_dataset, test_dataset = dsd100_dataset.get_datasets()
train_data_size, valid_data_size, test_data_size = dsd100_dataset.dataset_stat()

# separator model
separator = UnetConv2dResblock(2049, 87, (15, 3))
model = separator.get_model()
model.summary()


class ShowLearnintRate(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 10 == 0:
            print('\nEpoch %03d: Learning rate is %6.4f.' % (epoch, self.model.optimizer.lr.numpy()))


# callbacks: early-stopping, tensorboard
log_dir = "./logs/unet_conv2d_resblock/" + datetime.now().strftime("%Y%m%d_%H%M%S")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, verbose=True, patience=2),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ShowLearnintRate(),
]

# BEGIN TRAINING
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
              loss={'vocals': tf.keras.losses.MeanSquaredError(),
                    'bass': tf.keras.losses.MeanSquaredError(),
                    'drums': tf.keras.losses.MeanSquaredError(),
                    'other': tf.keras.losses.MeanSquaredError()})

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=valid_dataset,
                    steps_per_epoch=train_data_size // BATCH_SIZE,
                    validation_steps=valid_data_size // BATCH_SIZE,
                    callbacks=callbacks)

# save model
date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
current_file_path = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(current_file_path))
saved_model_dir = os.path.join(root, 'saved_model')
saved_model_name = os.path.join(
    saved_model_dir, 'unet_conv2d_resblock?time={}.h5'.format(date_time))
model.save(saved_model_name)
