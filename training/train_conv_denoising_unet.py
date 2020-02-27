import tensorflow as tf
from models.conv_denoising_unet import ConvDenoisingUnet
from training.train_loop import TrainLoop


separator = ConvDenoisingUnet(1025, 100)
optimizer = tf.keras.optimizers.Adam(lr=0.0002)
training = TrainLoop(separator, batch_size=32, max_epochs=100, optimzier=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=10, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
