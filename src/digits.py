import struct

import numpy as np
import matplotlib.pyplot as plt

from network import Network

size = (784, 28, 28, 10)
nn = Network(size)

def load_labels(path):
    with open("../mnist/" + path, 'rb') as file:
        magic, length = struct.unpack('>II', file.read(8))
        assert magic == 2049
        array = np.fromfile(file, dtype=np.dtype(np.uint8))
    return array.reshape(length)

def load_images(path):
    with open("../mnist/" + path, 'rb') as file:
        magic, length, rows, cols = struct.unpack('>IIII', file.read(16))
        assert magic == 2051
        array = np.fromfile(file, dtype=np.dtype(np.uint8))
    return array.reshape(length, rows, cols) / 255

train_images = load_images("train-images.idx3-ubyte")
train_labels = load_labels("train-labels.idx1-ubyte")
test_images = load_images("t10k-images.idx3-ubyte")
test_labels = load_labels("t10k-labels.idx1-ubyte")

def plot_image(image, title=None):
    plt.imshow(image)
    plt.set_cmap("binary")
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_error(error):
    plt.plot(error)
    plt.title("Error during training")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

def train(images, labels):
    print("Training started.")
    training_data = []
    for i in range(len(labels)):
        inputs = images[i].reshape((-1, 1))
        outputs = np.array([1 if j == labels[i] else 0 for j in range(size[-1])]).reshape((-1, 1))
        training_data.append((inputs, outputs))
    error = nn.train(training_data)
    plot_error(error)
    print("Training finished.")

def test(images, labels):
    right = 0
    print("Testing started.")
    for i, image in enumerate(images):
        outputs = nn.forward(image)
        guess = (-1, -1)
        for j, value in enumerate(outputs):
            guess = max(guess, (value, j))
        guess = guess[1]
        if guess != labels[i]:
            plot_image(image, title=f"{guess}, ({labels[i]})")
        else:
            right += 1
    print(f"The network classified {right}/{len(labels)} ({right / len(labels) * 100}%) correctly.")
    print("Testing finished.")

train(train_images, train_labels)
test(test_images, test_labels)
