"""Network
==========

This module contains the Network class that can be used to create a neural network.
"""

import math

import numpy as np

class Network:
    """A neural network that can be trained for a given task.

    After successful training the network should have picked up the underlying trend in the
    training data and should be able to generalize it to cases not seen before.

    Attributes:
        size (tuple): Information regarding the amount of layers and neurons in the network.
        values (list): Values of every neuron in the network.
        weights (list): Weights of every neuron in the network.
        biases (list): Biases of every neuron in the network.
    """

    def __init__(self, size):
        """Create a new neural network.

        Args:
            size (Iterable[int]): The length of this parameter is the amount of layers in
              the network and the elements are the amounts of neurons in each layer respectively.

        Raises:
            ValueError: If the amount of layers is under 2 or
              the amount of neurons in a layer is under 1.
        """
        if len(size) < 2 or not all(layer_size > 0 for layer_size in size):
            raise ValueError("illegal size for the neural network")
        self.size = tuple(size)
        self.values = [None for _ in range(len(size))]
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
            ndarray: The weights matrix for the current layer.
        """
        x = math.sqrt(6 / (size + old_size))
        weights = np.random.uniform(-x, x, (size, old_size))
        return weights

    def sigmoid(self, array):
        """Apply sigmoid function element-wise to a vector.

        Args:
            array (ndarray): Array to be worked on.

        Returns:
            ndarray: Array like the original array but sigmoid applied to the elements.
        """
        new_array = np.zeros_like(array)
        for i, x in enumerate(array):
            new_array[i] = 1 / (1 + math.exp(-x[0]))
        return new_array

    def forward(self, inputs):
        """Forward inputs through the network and return outputs.

        Args:
            inputs (Iterable[float]): The inputs to be forwarded. Should be floats between 0 and 1.
              The length of this parameter must be equal to the length of the input layer.

        Returns:
            ndarray: Values of the output neurons after the forwarding.
        """
        self.values[0] = np.reshape(inputs, (self.size[0], 1))
        for i in range(1, len(self.size)):
            vector = np.matmul(self.weights[i], self.values[i - 1]) + self.biases[i]
            self.values[i] = self.sigmoid(vector)
        return self.values[-1]

    def __repr__(self):
        return f"Network{self.size}"
