import tensorflow as tf
from models.conv_resblock_denoising_unet import ConvResblockDenoisingUnet
from training.train_loop import TrainLoop


separator = ConvResblockDenoisingUnet(1025, 100)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
training = TrainLoop(separator, batch_size=16, max_epochs=50, optimzier=optimizer)


if __name__ == '__main__':
    history = training.compile_and_fit(validation_freq=10, verbose=2)
    training.evaluate()
    training.save_model()
    training.save_weights()
