from os import path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.engine.saving import load_model
from tensorflow.python.keras.layers import Dense


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def autoencoder():
    input_image = Input(shape=(784,))
    # Encoder
    encoder = Dense(units=128, activation='relu')(input_image)
    encoder = Dense(units=64, activation='relu')(encoder)
    encoder = Dense(units=32, activation='relu')(encoder)

    # Decoder
    decoder = Dense(units=64, activation='relu')(encoder)
    decoder = Dense(units=128, activation='relu')(decoder)
    decoder = Dense(units=784, activation='sigmoid')(decoder)

    enc = Model(
        input_image, encoder
    )

    autoenc = Model(
        input_image, decoder
    )

    autoenc.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    return enc, autoenc


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.reshape(x_train, (60000, 784))
    x_test = np.reshape(x_test, (10000, 784))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    encoder, autoenc = autoencoder()
    lossHistory = LossHistory()
    autoenc.summary()
    epochs = 100
    autoenc.fit(x_train, x_train, epochs= epochs, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[lossHistory])

    autoencoded_images = autoenc.predict(x_test[0:10]).reshape((10, 28, 28))
    # encoded_images = encoder.predict(x_test[0:10]).reshape((10, 28, 28))
    for i in range(0, 10):
        figure = plt.figure()
        im1 = figure.add_subplot(2, 1, 1)
        im1.imshow(x_test[i].reshape((28, 28)) * 255)
        # im1.title("image pas encodée")
        im2 = figure.add_subplot(2, 1, 2)
        im2.imshow(autoencoded_images[i].reshape((28, 28)) * 255)
        #   plt.imshow(encoded_images[i] * 255)
        # im2.title("image encodée ")
        plt.show()

    plt.plot(lossHistory.losses)
    plt.show()
    encoder.save("encoder_model")
    autoenc.save("autoencoder_model")
