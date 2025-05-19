"""network.py

This module contains the Network class that can be used to create a neural network.
"""

import math

import numpy as np

class Network:
    """A neural network that can be trained for a given task.

    After successful training the network should have picked up the underlying trend in the
    training data and should be able to generalize it to cases not seen before.

    Attributes:
        size (tuple): AContains integers whose length is the amount of layers in the network
          and the elements are the amounts of neurons in each layer respectively.
          This is also the only parameter for the constructor.
        values (list): Contains values of every neuron in the network.
        weights (list): Contains weights of every neuron in the network.
        biases (list): Contains biases of every neuron in the network.
    """

    def __init__(self, *size):
        if len(size) < 2 or not all(layer_size > 0 for layer_size in size):
            raise ValueError("illegal size for the neural network")
        self.size = tuple(size)
        self.values = [np.zeros((self.size[i], 1)) for i in range(len(self.size))]
        self.weights = [None]
        self.biases = [None]
        for i in range(1, len(self.size)):
            self.biases.append(np.zeros((self.size[i], 1)))
            self.weights.append(self.xavier(self.size[i], self.size[i - 1]))

    def xavier(self, size, old_size):
        """Uniform xavier weight initialization for layers using the sigmoid activation funcion.

        Args:
            size (int): The size of the current layer.
            old_size (int): The size of the previous layer.

        Returns:
            NDArray: The weights matrix for the current layer.
        """

        x = math.sqrt(6 / (size + old_size))
        weights = np.random.uniform(-x, x, (size, old_size))
        return weights

    def __repr__(self):
        return f"Network{self.size}"
