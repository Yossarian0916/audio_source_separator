#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
from models.conv_denoising_unet import ConvDenoisingUnet
from training.make_dataset import DSD100Dataset


tf.get_logger().setLevel('ERROR')
# hyper-parameter
BATCH_SIZE = 32

# load dataset
dsd100_dataset = DSD100Dataset(batch_size=BATCH_SIZE)
train_dataset, valid_dataset, test_dataset = dsd100_dataset.get_datasets()
train_data_size, valid_data_size, test_data_size = dsd100_dataset.dataset_stat()

# separator model
separator = ConvDenoisingUnet(1025, 100)
model = separator.get_model()
model.summary()

# callbacks: early-stopping, tensorboard
current_path = os.path.abspath(__file__)
training_path = os.path.dirname(current_path)
log_dir = os.path.join(training_path, "./logs/conv_denoising_unet/" + datetime.now().strftime("%Y%m%d_%H%M"))
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=1),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]

# BEGIN TRAINING
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003),
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
                    validation_freq=5)

model.evaluate(test_dataset, steps=test_data_size // BATCH_SIZE)
# save model
date_time = datetime.now().strftime("%Y%m%d_%H%M")
current_file_path = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(current_file_path))
saved_model_dir = os.path.join(root, 'saved_model')
saved_model_name = os.path.join(saved_model_dir, 'conv_denoising_unet?time={}.h5'.format(date_time))
model.save(saved_model_name)
print("\nModel Saved Successful!")
