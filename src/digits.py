import struct
import math

import numpy as np
import matplotlib.pyplot as plt

from network import Network

def load_labels(path):
    with open(path, 'rb') as file:
        magic, length = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError(f"magic number mismatch: got {magic}, expected 2049")
        array = np.fromfile(file, dtype=np.dtype(np.uint8))
    return array.reshape(length)

def load_images(path):
    with open(path, 'rb') as file:
        magic, length, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError(f"magic number mismatch: got {magic}, expected 2051")
        array = np.fromfile(file, dtype=np.dtype(np.uint8))
    return array.reshape(length, rows, cols) / 255

def plot_images(images):
    for image in images:
        plt.imshow(image[0])
        plt.set_cmap("binary")
        plt.colorbar()
        plt.title(image[1])
        plt.show()

def plot_error(error):
    plt.plot(range(1, len(error) + 1), error)
    ticks = list(range(0, len(error) + 1, math.ceil(len(error) / 10)))[1:]
    if ticks[0] != 1:
        ticks = [1] + ticks
    if ticks[-1] != len(error):
        ticks = ticks + [len(error)]
    plt.xticks(ticks)
    plt.title("Error during training")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

def train(nn, images, labels):
    print("Training started.")
    training_data = []
    for i in range(len(labels)):
        inputs = images[i]
        outputs = np.array([1 if j == labels[i] else 0 for j in range(nn.shape[-1])])
        training_data.append((inputs.reshape((-1, 1)), outputs.reshape((-1, 1))))
    errors = []
    print("1/2")
    errors += nn.train(training_data, 10, batch_size=1, learning_rate=1)
    print("2/2")
    errors += nn.train(training_data, 10, batch_size=10, learning_rate=1)
    print("Training finished.")
    return errors

def test(nn, images, labels):
    print("Testing started.")
    right = []
    wrong = []
    for i, image in enumerate(images):
        label = labels[i]
        outputs = nn.forward(image)
        guess = (-1, -1)
        for j, value in enumerate(outputs):
            guess = max(guess, (value, j))
        guess = guess[1]
        if guess != label:
            wrong.append((image, f"Guess: {guess}, Actually: {label}"))
        else:
            right.append((image, f"{label}"))
    percent = len(right) / len(labels) * 100
    print(f"The network classified {len(right)}/{len(labels)} ({percent:0.2f}%) correctly.")
    print("Testing finished.")
    return wrong, right

def main():
    nn = Network((784, 28, 28, 10))

    path = "../mnist/"
    train_images = load_images(path + "train-images.idx3-ubyte")
    train_labels = load_labels(path + "train-labels.idx1-ubyte")
    test_images = load_images(path + "t10k-images.idx3-ubyte")
    test_labels = load_labels(path + "t10k-labels.idx1-ubyte")

    print("Before training:")
    test(nn, test_images, test_labels)
    print()
    error = train(nn, train_images, train_labels)
    print("Plotting error.")
    plot_error(error)
    print()
    print("After training:")
    results = test(nn, test_images, test_labels)
    print()
    print("Plotting images classified incorrectly.")
    print("To show the next image, close the previous one. Quit with ctrl + c")
    plot_images(results[0])


if __name__ == "__main__":
    main()
