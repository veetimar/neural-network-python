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

    def __init__(self, shape):
        """Create a new neural network.

        Args:
            size (Iterable[int]): The length of this parameter is the amount of layers in
              the network and the elements are the amounts of neurons in each layer respectively.

        Raises:
            ValueError: If the amount of layers is under 2 or
              the amount of neurons in a layer is under 1.
        """
        if len(shape) < 2 or not all(layer_size > 0 for layer_size in shape):
            raise ValueError("illegal size for the neural network")
        self.shape = tuple(shape)
        self.values = [None for _ in range(len(shape))]
        self.weights = [None]
        self.biases = [None]
        for i in range(1, len(self.shape)):
            self.biases.append(np.zeros((self.shape[i], 1)))
            self.weights.append(self.xavier(self.shape[i], self.shape[i - 1]))

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
        return 1 / (1 + np.exp(-array))

    def sigmoid_derivative(self, array):
        """Apply the derivative of sigmoid function element-wise to a vector.

        Args:
            array (ndarray): Array to be worked on.

        Returns:
            ndarray: Array like the original array but the derivative applied to the elements.
        """
        x = self.sigmoid(array)
        new_array = x * (1 - x)
        return new_array

    def mean_squared(self, outputs, expected):
        """Calculate the mean squared error.

        Args:
            outputs (ndarray): Outputs of the neural network.
            expected (ndarray): What the outputs should have been.

        Returns:
            float: The error value.
        """
        x = outputs - expected
        return np.matmul(x.T, x) / len(x)

    def mean_squared_gradient(self, outputs, expected):
        """Calculate the gradient of mean squared error with respect to outputs.

        Args:
            outputs (ndarray): Outputs of the neural network.
            expected (ndarray): What the outputs should have been.

        Returns:
            ndarray: The gradient.
        """
        x = outputs - expected
        return 2 * x / len(x)

    def forward(self, inputs):
        """Forward inputs through the network and return outputs.

        Args:
            inputs (Iterable[float]): The inputs to be forwarded. Should be floats between 0 and 1.
              The length of this parameter must be equal to the length of the input layer.

        Returns:
            ndarray: Values of the output neurons after the forwarding.
        """
        self.values[0] = np.reshape(inputs, (self.shape[0], 1))
        for i in range(1, len(self.shape)):
            vector = np.matmul(self.weights[i], self.values[i - 1]) + self.biases[i]
            self.values[i] = self.sigmoid(vector)
        return self.values[-1]

    def save(self, path=""):
        """Save current weights and biases to the disk.

        Args:
            path (str, optional): Path to the directory where the files will be saved.
              Path must exist already. Defaults to "".
        """
        if path and not path.endswith("/"):
            path += "/"
        np.savez_compressed(path + "weights.npz", *self.weights[1:])
        np.savez_compressed(path + "biases.npz", *self.biases[1:])

    def load(self, path=""):
        """Load current weights and biases from the disk.

        Args:
            path (str, optional): Path to the directory containing the files. Defaults to "".

        Raises:
            ValueError: If the shape of loaded weights and biases do not fit to the network.
        """
        if path and not path.endswith("/"):
            path += "/"
        weights = [None]
        biases = [None]
        weights += np.load(path + "weights.npz").values()
        biases += np.load(path + "biases.npz").values()
        if len(weights) != len(self.shape) or len(biases) != len(self.shape):
            raise ValueError("loaded weights or biases do not fit to the network")
        shape = self.shape
        for i in range(1, len(self.shape)):
            if weights[i].shape != (shape[i], shape[i - 1]) or biases[i].shape != (shape[i], 1):
                raise ValueError("loaded weights or biases do not fit to the network")
        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return f"Network{self.shape}"
