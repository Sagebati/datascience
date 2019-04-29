from tensorflow import keras

from auto_encoder import trained_encoder


def get_encoded_data():
    model = trained_encoder()
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((60000, 784))
    x_train = x_train.astype('float32') / 255
    x_reshaped = x_test.reshape((10000, 784))
    x_reshaped = x_reshaped.astype('float32') / 255
    x_train_encoded = model.predict(x_train)
    x_test_encoded = model.predict(x_reshaped)

    return (x_train_encoded, y_train), (x_test_encoded, y_test)
