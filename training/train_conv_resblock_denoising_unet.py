import tensorflow as tf
from models.conv_resblock_denoising_unet import ConvResblockDenoisingUnet
from training.train_loop import TrainLoop


separator = ConvResblockDenoisingUnet(1025, 100)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0002, decay_steps=4000, decay_rate=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
training = TrainLoop(separator, batch_size=16, max_epochs=20, optimizer=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=5, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
