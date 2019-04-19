import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    plt.show()
