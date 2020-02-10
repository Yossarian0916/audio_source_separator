#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
from models.autoencoder_conv2d import AutoencoderConv2d
from training.make_dataset import DSD100Dataset


tf.get_logger().setLevel('ERROR')
# hyper-parameter
BATCH_SIZE = 8

# load dataset
dsd100_dataset = DSD100Dataset(batch_size=BATCH_SIZE)
train_dataset, valid_dataset, test_dataset = dsd100_dataset.get_datasets()
train_data_size, valid_data_size, test_data_size = dsd100_dataset.dataset_stat()

# separator model
separator = AutoencoderConv2d(2049, 87, (3, 3))
model = separator.get_model()
model.summary()

# callbacks: early-stopping, tensorboard
log_dir = "./logs/autoencoder_conv2d/" + datetime.now().strftime("%Y%m%d_%H%M")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=1),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]

# BEGIN TRAINING
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss={'vocals': tf.keras.losses.MeanSquaredError(),
                    'bass': tf.keras.losses.MeanSquaredError(),
                    'drums': tf.keras.losses.MeanSquaredError(),
                    'other': tf.keras.losses.MeanSquaredError()})

history = model.fit(train_dataset,
                    epochs=50,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_dataset,
                    steps_per_epoch=train_data_size // BATCH_SIZE,
                    validation_steps=valid_data_size // BATCH_SIZE,
                    validation_freq=10,
                    max_queue_size=10,
                    workers=8,
                    use_multiprocessing=True)

# save model
date_time = datetime.now().strftime("%Y%m%d_%H%M")
current_file_path = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(current_file_path))
saved_model_dir = os.path.join(root, 'saved_model')
saved_model_name = os.path.join(saved_model_dir, 'autoencoder_conv2d_layerNormalization?time={}.h5'.format(date_time))
model.save(saved_model_name)
