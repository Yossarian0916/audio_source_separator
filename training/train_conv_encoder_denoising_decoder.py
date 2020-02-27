import tensorflow as tf
from models.conv_encoder_denoising_decoder import ConvEncoderDenoisingDecoder
from training.train_loop import TrainLoop


separator = ConvEncoderDenoisingDecoder(1025, 100)
optimizer = tf.keras.optimizers.Adam(lr=0.0003)
training = TrainLoop(separator, batch_size=32, max_epochs=200, optimizer=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=10, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
