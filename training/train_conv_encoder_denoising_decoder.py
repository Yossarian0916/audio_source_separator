import tensorflow as tf
from models.conv_encoder_denoising_decoder import ConvEncoderDenoisingDecoder
from training.train_loop import TrainLoop


separator = ConvEncoderDenoisingDecoder(1025, 100)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005, decay_steps=4000, decay_rate=0.5, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
training = TrainLoop(separator, batch_size=32, max_epochs=50, optimizer=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=10, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
