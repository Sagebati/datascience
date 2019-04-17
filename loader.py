import numpy as np


def load_labels(path):
    with open(path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        number_of_items = int.from_bytes(f.read(4), byteorder='big')
        buf = f.read(number_of_items)
    data = np.frombuffer(buf, dtype=np.uint8)
    return data


def load_images(path):
    with open(path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        number_of_items = int.from_bytes(f.read(4), byteorder='big')
        nb_columns = int.from_bytes(f.read(4), byteorder='big')
        nb_rows = int.from_bytes(f.read(4), byteorder='big')
        buf = f.read(number_of_items * nb_columns * nb_rows)
    return np.frombuffer(buf, dtype=np.uint8).reshape(number_of_items, nb_rows, nb_columns, 1)
