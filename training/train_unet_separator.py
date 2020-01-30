import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os
import sys
from models.unet_separator import UnetSeparator
from training.make_dataset import DSD100Dataset


# load dataset
BATCH_SIZE = 8
dsd100_dataset = DSD100Dataset(batch_size=BATCH_SIZE)
train_dataset, valid_dataset, test_dataset = dsd100_dataset.get_datasets()
TRAIN_DATA_SIZE, VAL_DATA_SIZE, TEST_DATA_SIZE = dsd100_dataset.dataset_stat()

# separator model
separator = UnetSeparator(2049, 87)
model = separator.get_model()
model.summary()


class ShowLearnintRate(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 10 == 0:
            print('\nEpoch %05d: Learning rate is %6.4f.' %
                  (epoch, self.model.optimizer.lr.numpy()))


def decay(epoch):
    if epoch < 50:
        return 0.3
    else:
        return 0.03


log_dir = "./logs/unet_separator/" + \
    datetime.now().strftime("%Y%m%d_%H%M%S")

# callbacks: early-stopping, tensorboard
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-3, verbose=True, patience=20),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.LearningRateScheduler(decay),
    ShowLearnintRate(),
]

# BEGIN TRAINING
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.3, momentum=0.9, nesterov=True),
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
current_file_path = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(current_file_path))
saved_model_dir = os.path.join(root, 'saved_model')
saved_model_name = os.path.join(
    saved_model_dir, 'unet_separator?time={}.h5'.format(date_time))
model.save(saved_model_name)
