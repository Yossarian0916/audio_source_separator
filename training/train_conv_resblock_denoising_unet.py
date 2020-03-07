import tensorflow as tf
from models.conv_resblock_denoising_unet import ConvResblockDenoisingUnet
from training.train_loop import TrainLoop


separator = ConvResblockDenoisingUnet(1025, 100)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0003,
        decay_steps=30000,
        decay_rate=0.1,
        staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
training = TrainLoop(separator, batch_size=4, max_epochs=10, optimizer=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=5, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
