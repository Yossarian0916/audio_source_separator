import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from training.make_dataset import DSD100Dataset
from utils import module_path


class TrainLoop:
    def __init__(self, separator_model, batch_size=32, max_epochs=100, optimizer=None):
        self.model = separator_model.get_model()
        self.model_name = separator_model.get_name()
        self.batch_size = batch_size
        self.optimizer = None
        self.max_epochs = max_epochs
        self.timestamp = self.get_datetime()

    def prepare_dataset(self):
        dsd100_dataset = DSD100Dataset(batch_size=self.batch_size)
        (self.train_dataset, self.valid_dataset, self.test_dataset) = dsd100_dataset.get_datasets()
        (self.train_data_size, self.valid_data_size, self.test_data_size) = dsd100_dataset.dataset_stat()

    def get_callbacks(self):
        # callbacks: early-stopping, tensorboard
        training_module_path = module_path.get_training_path()
        log_dir = os.path.join(training_module_path, "logs", self.model_name, self.timestamp)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=1),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        ]
        return callbacks

    def compile_and_fit(self, validation_freq=10, verbose=2, new_callbacks=None, ):
        tf.get_logger().setLevel('ERROR')
        # train and validation datasets
        self.prepare_dataset()
        # optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(lr=0.0003)
        # loss functions
        loss = {'vocals': tf.keras.losses.MeanSquaredError(),
                'bass': tf.keras.losses.MeanSquaredError(),
                'drums': tf.keras.losses.MeanSquaredError(),
                'other': tf.keras.losses.MeanSquaredError()}
        # callbacks
        if new_callbacks is not None:
            callbacks = self.get_callbacks() + new_callbacks
        else:
            callbacks = self.get_callbacks()
        # compile model
        self.model.compile(optimizer=self.optimizer, loss=loss)
        self.model.summary()
        # training loop
        history = self.model.fit(self.train_dataset,
                                 epochs=self.max_epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=self.valid_dataset,
                                 steps_per_epoch=self.train_data_size // self.batch_size,
                                 validation_steps=self.valid_data_size // self.batch_size,
                                 validation_freq=validation_freq)
        return history

    def evaluate(self):
        self.model.evaluate(self.test_dataset, steps=self.test_data_size // self.batch_size, verbose=2)

    def save_model(self):
        # save model
        saved_model_dir = module_path.get_saved_model_path()
        saved_model_name = os.path.join(saved_model_dir, self.model_name+'?time={}.h5'.format(self.timestamp))
        self.model.save(saved_model_name)
        print("\nModel Saved Successful!")

    def save_weights(self):
        # save model weights only
        saved_model_dir = module_path.get_saved_model_path()
        saved_weight_dir = os.path.join(saved_model_dir, 'weight_checkpoints')
        saved_weights_name = os.path.join(saved_weight_dir, self.model_name+'?time={}.h5'.format(self.timestamp))
        self.model.save_weights(saved_weights_name)
        print("\nModel Weights Saved Successful!")

    def get_datetime(self):
        return datetime.now().strftime("%Y%m%d_%H%M")
