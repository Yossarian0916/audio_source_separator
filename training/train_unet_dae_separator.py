import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
import sys
from utils.helper import get_filenames
from utils.dataset import tfrecord2dataset
from models.unet_dae import UNet_Autoencoder


# load dataset
BATCH_SIZE = 64
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = os.path.join(root, 'data')
# tfrecords
dsd100_train_dir = os.path.join(data_dir, 'dsd100_train_tfrecords')
dsd100_train_tfrecords = get_filenames(dsd100_train_dir+'/*')
dsd100_test_dir = os.path.join(data_dir, 'dsd100_test_tfrecords')
dsd100_test_tfrecords = get_filenames(dsd100_test_dir+'/*')
# training dataset
train_tfrecords = dsd100_train_tfrecords + \
    dsd100_test_tfrecords[:len(dsd100_test_tfrecords)//2]
train_dataset = tfrecord2dataset(train_tfrecords, batch_size=BATCH_SIZE)
# validation dataset
valid_tfrecords = dsd100_test_tfrecords[len(dsd100_test_tfrecords)//2:]
valid_dataset = tfrecord2dataset(valid_tfrecords, batch_size=BATCH_SIZE)

TRAIN_DATA_SIZE = len(train_tfrecords)
VAL_DATA_SIZE = len(valid_tfrecords)

# separator model
separator = UNet_Autoencoder(2049, 87)
model = separator.get_model()

# callbacks: early-stopping, tensorboard
log_dir = "./logs/unet_wide&deep_separator/" + \
    datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-3, verbose=True, patience=10),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
]


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss={'vocals': tf.keras.losses.MeanSquaredError(),
                    'bass': tf.keras.losses.MeanSquaredError(),
                    'drums': tf.keras.losses.MeanSquaredError(),
                    'other': tf.keras.losses.MeanSquaredError()})

history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=valid_dataset,
                    steps_per_epoch=TRAIN_DATA_SIZE // BATCH_SIZE,
                    validation_steps=VAL_DATA_SIZE // BATCH_SIZE,
                    callbacks=callbacks)

# save model
date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
saved_model_dir = os.path.join(root, 'saved_model')
saved_model_name = os.path.join(
    saved_model_dir, 'unet_dae_separator?time={}.h5'.format(date_time))
model.save(saved_model_name)
