"""Digits
=========

This module contains the Digits class that can be used to test the Network module's ability to learn
and classify handwritten digits in the MNIST dataset.
"""

import struct
import math
import gzip

import numpy as np
import matplotlib.pyplot as plt

from network import Network

class Digits:
    """User interface for testing and training the Network class on MNIST handwritten digits.

    Contains methods for training the network to recognize handwritten digits as well as to test how
    well the network is able to identify them using the MNIST handwritten digits dataset.

    Attributes:
        run (bool): Controls whether the program should keep running.
        nn (Network): Neural network used to recognize the digits.
        training_data (list[tuple]): Data used to train the network.
        testing_data (list[tuple]): Data used to test the network.
    """

    def __init__(self, path=""):
        """Create a new user interface.

        Args:
            path (str, optional): Path to the directory containing the MNIST files. Defaults to "".
        """
        self.run = True
        self.nn = Network((784, 56, 28, 10))
        if path and not path.endswith("/"):
            path += "/"
        train_images = self.load_images(path + "train-images-idx3-ubyte.gz")
        train_labels = self.load_labels(path + "train-labels-idx1-ubyte.gz")
        test_images = self.load_images(path + "t10k-images-idx3-ubyte.gz")
        test_labels = self.load_labels(path + "t10k-labels-idx1-ubyte.gz")
        self.training_data = []
        self.testing_data = []
        for i, image in enumerate(train_images):
            inputs = image
            outputs = np.array([1 if j == train_labels[i] else 0 for j in range(self.nn.shape[-1])])
            self.training_data.append((inputs.reshape((-1, 1)), outputs.reshape((-1, 1))))
        for i, image in enumerate(test_images):
            inputs = image
            label = test_labels[i]
            self.testing_data.append((inputs, label))

    @classmethod
    def load_labels(cls, path):
        """Load the labels for the digits from a gz compressed file.

        Args:
            path (str): Path to the file.

        Raises:
            ValueError: If the magic number read from the file is not equal to the expected number.

        Returns:
            ndarray: The labels.
        """
        with gzip.open(path, 'rb') as file:
            magic, length = struct.unpack('>II', file.read(8))
            ex = 2049
            if magic != ex:
                raise ValueError(f"magic number mismatch: got {magic}, expected {ex}")
            buffer = file.read(length)
            array = np.frombuffer(buffer, dtype=np.uint8)
        return array

    @classmethod
    def load_images(cls, path):
        """Load the handwritten digits from a gz compressed file.

        Args:
            path (str): Path to the file.

        Raises:
            ValueError: If the magic number read from the file is not equal to the expected number.

        Returns:
            ndarray: The images.
        """
        with gzip.open(path, 'rb') as file:
            magic, length, rows, cols = struct.unpack('>IIII', file.read(16))
            ex = 2051
            if magic != ex:
                raise ValueError(f"magic number mismatch: got {magic}, expected {ex}")
            buffer = file.read(length * rows * cols)
            array = np.frombuffer(buffer, dtype=np.uint8)
        return array.reshape(length, rows, cols) / 255

    @classmethod
    def plot_images(cls, images):
        """Plot a sequence of images.

        Args:
            images (list[ndarray]): Images to be plotted.
        """
        with plt.ion():
            i = 0
            while True:
                plt.imshow(images[i][0])
                plt.title(images[i][1])
                plt.set_cmap("binary")
                plt.colorbar()
                accepted = ["d", "f", ""]
                prompt = "\"f\" to show next, \"d\" to show previous, blank to exit"
                action = cls.ask_string(prompt, accepted)
                if action == "d":
                    if i > 0:
                        i -= 1
                elif action == "f":
                    if i < len(images) - 1:
                        i += 1
                elif action == "":
                    plt.close("all")
                    break
                plt.close()

    @classmethod
    def plot_error(cls, errors):
        """Plot the average error of each epoch during training.

        Args:
            error (list[float]): Error to be plotted.
        """
        plt.plot(range(1, len(errors) + 1), errors, marker="o")
        ticks = list(range(0, len(errors) + 1, math.ceil(len(errors) / 10)))[1:]
        if ticks[0] != 1:
            ticks = [1] + ticks
        if ticks[-1] != len(errors):
            ticks = ticks + [len(errors)]
        plt.xticks(ticks)
        plt.title("Average error during training")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()

    @classmethod
    def ask_string(cls, prompt, accepted):
        """Ask the user for string input.

        Try again if the user input is not in the list of accepted strings.

        Args:
            prompt (str): Prompt to be displayed to the user.
            accepted (list[str]): Valid strings that are accepted as input.

        Returns:
            str: Accepted user input.
        """
        accepted = [text.casefold() for text in accepted]
        while True:
            result = input(prompt + ": ").casefold()
            if result in accepted:
                return result
            print("Trying again.")

    @classmethod
    def ask_input(cls, prompt, datatype, default):
        """Ask the user for input casted into specific datatype.

        Try again if the user input is not castable to the datatype.

        Args:
            prompt (str): Prompt to be displayed to the user.
            datatype (class): Datatype the input is casted into.
            default (Any): Default value (used when the user inputs empty string).

        Returns:
            Any: Accepted user input or default value.
        """
        while True:
            result = input(prompt + f" ({default}): ")
            if not result:
                return default
            try:
                return datatype(result)
            except ValueError:
                pass
            print("Trying again.")

    def train(self):
        """Train the network to recognize handwritten digits and plot the error afterwards.

        Ask the user for preferred hyperparameters and suggest the default values.
        """
        epochs = self.ask_input("How many epochs", int, 10)
        batch_size = self.ask_input("Batch size", int, 1)
        learning_rate = self.ask_input("Learning rate", float, 0.1)
        print("Training started, please wait.")
        errors = self.nn.train(self.training_data, epochs, batch_size, learning_rate, True)
        print("Training finished.")
        print("Plotting error, close the figure to continue.")
        self.plot_error(errors)

    def test(self):
        """Test how well the network is able to recognize handwritten digits.

        Offer to plot images classified correctly or incorrectly afterwards.
        """
        print("Testing started.")
        right = []
        wrong = []
        for image, label in self.testing_data:
            outputs = self.nn.forward(image)
            guess = np.argmax(outputs)
            if guess != label:
                wrong.append((image, f"Guess: {guess}, Actually: {label}"))
            else:
                right.append((image, f"{label}"))
        data_length = len(self.testing_data)
        percent = len(right) / data_length * 100
        print(f"The network classified {len(right)}/{data_length} ({percent:0.2f}%) correctly.")
        print("Testing finished.")
        prompt = "\"w\" to show images classified wrong, \"r\" to show images classified right"
        prompt += ", blank to skip"
        while True:
            action = self.ask_string(prompt, ["w", "r", ""])
            if action == "w":
                self.plot_images(wrong)
            elif action == "r":
                self.plot_images(right)
            elif action == "":
                break

    def main(self):
        """The main method that starts the UI.
        """
        while self.run:
            accepted = ["test", "train", "exit", "save", "load"]
            action = self.ask_string("\nTest, train, exit, save, or load", accepted)
            if action == "test":
                self.test()
            elif action == "train":
                self.train()
            elif action == "exit":
                self.exit()
            elif action == "save":
                self.nn.save()
                print("Saved parameters to the disk.")
            elif action == "load":
                try:
                    self.nn.load()
                    print("Loaded parameters from the disk.")
                except FileNotFoundError as e:
                    print("File not found:")
                    print(e)

    def exit(self):
        """Quit the program.
        """
        self.run = False
        print("Terminating...")


if __name__ == "__main__":
    try:
        ui = Digits("mnist")
        ui.main()
    except KeyboardInterrupt:
        ui.exit()
