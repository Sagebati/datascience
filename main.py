from loader import load_images
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = load_images("mnist/t10k-images-idx3-ubyte")
    image = np.array(data[2]).squeeze()
    plt.imshow(image)
    plt.show()
