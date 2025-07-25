"""Network
==========

This module contains the Network class that can be used to create a neural network.
"""

import math
import random

import numpy as np

class Network:
    """A neural network that can be trained for a given task.

    After successful training the network should have picked up the underlying trend in the training
    data and should be able to generalize it to cases not seen before.

    Attributes:
        shape (tuple): Information regarding the number of layers and neurons in the network.
        values (list): Values of every neuron in the network.
        activations (list): Values but after applying the activation function.
        gradients (list): Used during backpropagation to speed up and simplify calculations.
        weights (list): Weights of every neuron in the network.
        biases (list): Biases of every neuron in the network.
        delta_weights (list): Cached weights used during backpropagation.
        delta_biases (list): Cached biases used during backpropagation.
    """

    def __init__(self, shape):
        """Create a new neural network.

        Args:
            shape (Sequence[int]): The length of this parameter is the number of layers in the
             network and the elements are the numbers of neurons in each layer respectively.

        Raises:
            ValueError: If the number of layers is under 2 or the number of neurons in a layer is
              under 1.
        """
        if len(shape) < 2 or not all(size > 0 for size in shape):
            raise ValueError("illegal shape for the neural network")
        self.shape = tuple(shape)
        self.values = [None for _ in range(len(shape))]
        self.activations = [None for _ in range(len(shape))]
        self.gradients = [None for _ in range(len(shape))]
        self.weights = [None]
        self.biases = [None]
        self.delta_weights = [None]
        self.delta_biases = [None]
        for i in range(1, len(self.shape)):
            if i != len(self.shape) - 1:
                self.weights.append(self.he(self.shape[i - 1], self.shape[i]))
            else:
                self.weights.append(self.glorot(self.shape[i - 1], self.shape[i], 0))
            self.biases.append(np.zeros((self.shape[i], 1)))
            self.delta_weights.append(np.zeros_like(self.weights[i]))
            self.delta_biases.append(np.zeros_like(self.biases[i]))

    @classmethod
    def glorot(cls, old_size, size, next_size):
        """Uniform Glorot (Xavier) weight initialization for layers with sigmoid-like activation
        function.

        Args:
            size (int): The size of the current layer.
            old_size (int): The size of the previous layer.
            next_size (int): The size of the next layer.

        Returns:
            ndarray: The weights matrix for the current layer.
        """
        x = math.sqrt(6 / (old_size + next_size))
        weights = np.random.uniform(-x, x, (size, old_size))
        return weights

    @classmethod
    def he(cls, old_size, size):
        """Normal He (Kaiming) weight initialization for layers with ReLU-like activation function.

        Args:
            size (int): The size of the current layer.
            old_size (int): The size of the previous layer.

        Returns:
            ndarray: The weights matrix for the current layer.
        """
        scale = math.sqrt(2 / old_size)
        weights = np.random.normal(0, scale, (size, old_size))
        return weights

    @classmethod
    def sigmoid(cls, array):
        """Apply the sigmoid function element-wise to a vector.

        Args:
            array (ndarray): The array to be worked on.

        Returns:
            ndarray: Array like the original array but sigmoid applied to the elements.
        """
        return 1 / (1 + np.exp(-array))

    @classmethod
    def sigmoid_derivative(cls, array):
        """Apply the derivative of sigmoid function element-wise to a vector.

        Args:
            array (ndarray): The array to be worked on.

        Returns:
            ndarray: Array like the original array but the derivative applied to the elements.
        """
        x = cls.sigmoid(array)
        return x * (1 - x)

    @classmethod
    def elu(cls, array):
        """Apply the ELU function element-wise to a vector.

        Args:
            array (ndarray): The array to be worked on.

        Returns:
            ndarray: Array like the original array but ELU applied to the elements.
        """
        return np.where(array >= 0, array, np.exp(array) - 1)

    @classmethod
    def elu_derivative(cls, array):
        """Apply the derivative of ELU function element-wise to a vector.

        Args:
            array (ndarray): The array to be worked on.

        Returns:
            ndarray: Array like the original array but the derivative applied to the elements.
        """
        return np.where(array >= 0, 1, np.exp(array))

    @classmethod
    def mean_squared(cls, outputs, expected):
        """Calculate the mean squared error.

        Args:
            outputs (ndarray): Outputs of the neural network.
            expected (ndarray): What the outputs should have been.

        Returns:
            float: The error value.
        """
        x = np.reshape(outputs, -1) - np.reshape(expected, -1)
        return float(np.dot(x, x) / len(x))

    @classmethod
    def mean_squared_gradient(cls, outputs, expected):
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
            inputs (Sequence[float]): The inputs to be forwarded. Should be floats between 0 and 1.
              The length of this parameter must be equal to the length of the input layer.

        Returns:
            ndarray: Values of the output neurons after the forwarding.
        """
        self.activations[0] = np.reshape(inputs, (self.shape[0], 1))
        for i in range(1, len(self.shape)):
            vector = self.weights[i] @ self.activations[i - 1] + self.biases[i]
            self.values[i] = vector
            if i != len(self.shape) - 1:
                self.activations[i] = self.elu(vector)
            else:
                self.activations[i] = self.sigmoid(vector)
        return self.activations[-1]

    def backward(self, inputs, expected):
        """Backpropagation. 

        Adjust the parameters of the network in such a way that the provided inputs will
        produce outputs more like the expected outputs. Run update afterwards to apply the changes.

        Args:
            inputs (Sequence[float]): Inputs to the network. Should be floats between 0 and 1. The
              length of this parameter must be equal to the length of the input layer.
            expected (Sequence[float]): The expected outputs from the passed inputs. Should be
              floats between 0 and 1. The length of this parameter must be equal to the length of
              the output layer.

        Returns:
            float: Mean squared error calculated from the actual outputs and expected outputs.
        """
        inputs = np.reshape(inputs, (self.shape[0], 1))
        expected = np.reshape(expected, (self.shape[-1], 1))
        outputs = self.forward(inputs)
        error = self.mean_squared(outputs, expected)
        self.gradients[-1] = self.mean_squared_gradient(outputs, expected)
        self.gradients[-1] *= self.sigmoid_derivative(self.values[-1])
        for i in range(len(self.shape) - 1, 0, -1):
            self.delta_biases[i] += self.gradients[i]
            self.delta_weights[i] += self.gradients[i] @ self.activations[i - 1].T
            if i != 1:
                self.gradients[i - 1] = self.weights[i].T @ self.gradients[i]
                self.gradients[i - 1] *= self.elu_derivative(self.values[i - 1])
        return error

    def train(self, training_data, epochs, batch_size=-1, learning_rate=0.1, out=False):
        """Train the neural network.

        Args:
            training_data (list[tuple]): Contains tuples containing input and output examples. The
              first element of the tuple is inputs to the network and the second element is expected
              outputs.
            epochs (int): Number of epochs (how many times the whole dataset is iterated through).
            batch_size (int, optional): Determines how many training examples are processed before
              updating the weights. Setting this to -1 will set batch size to maximum i.e. the
              length of the training data. Defaults to -1.
            learning_rate (float, optional): Determines how fast the network tries to learn. The
              bigger this value the bigger steps the network takes towards the target. Defaults to
              0.1.
            out (bool, optional): Controls whether the average error of each epoch epoch is also
              printed to the terminal during training.

        Raises:
            ValueError: If the given number of epochs is not positive.
            ValueError: If the given batch size is not positive (-1 allowed) or is bigger than the
              length of the training data.
            ValueError: If the length of a training example is not 2 (contains something else than
              inputs and expected outputs).

        Returns:
            list[float]: Average error of each epoch in order from first to last.
        """
        if epochs < 1:
            raise ValueError("illegal number of epochs")
        if batch_size == -1:
            batch_size = len(training_data)
        if not 0 < batch_size <= len(training_data):
            raise ValueError("illegal batch size")
        if batch_size != len(training_data):
            random.shuffle(training_data)
        learning_rate /= batch_size
        errors = []
        for i in range(epochs):
            epoch_error = 0
            for j, sample in enumerate(training_data):
                if j > 0 and j % batch_size == 0:
                    self.update(learning_rate)
                if len(sample) != 2:
                    msg = "illegal structure in training data"
                    msg += f": expected sample length 2, got {len(sample)}"
                    raise ValueError(msg)
                epoch_error += self.backward(sample[0], sample[1])
            self.update(learning_rate)
            epoch_error /= len(training_data)
            if out:
                print(f"Epoch {i + 1} average error: {epoch_error}")
            errors.append(epoch_error)
        return errors

    def update(self, learning_rate):
        """Update the parameters and clear the delta variables.

        Args:
            learning_rate (float): Determines how fast the network tries to learn. The bigger this
              value the bigger steps the network takes towards the target.
        """
        for i in range(1, len(self.shape)):
            self.weights[i] -= learning_rate * self.delta_weights[i]
            self.biases[i] -= learning_rate * self.delta_biases[i]
            self.delta_weights[i].fill(0)
            self.delta_biases[i].fill(0)

    def save(self, path=""):
        """Save current parameters to the disk.

        Args:
            path (str, optional): Path to the directory where the files will be saved. Path must
              exist already. Defaults to "".
        """
        if path and not path.endswith("/"):
            path += "/"
        np.savez_compressed(path + "weights.npz", *self.weights[1:])
        np.savez_compressed(path + "biases.npz", *self.biases[1:])

    def load(self, path=""):
        """Load the parameters from the disk.

        Args:
            path (str, optional): Path to the directory containing the files. Defaults to "".

        Raises:
            ValueError: If the shape of loaded parameters do not fit the network.
        """
        if path and not path.endswith("/"):
            path += "/"
        weights = [None]
        biases = [None]
        weights += np.load(path + "weights.npz").values()
        biases += np.load(path + "biases.npz").values()
        if len(weights) != len(self.shape) or len(biases) != len(self.shape):
            msg = "loaded parameters do not fit the network"
            msg += f": expected parameters suitable for network shape {self.shape}"
            raise ValueError(msg)
        shape = self.shape
        for i in range(1, len(self.shape)):
            if weights[i].shape != (shape[i], shape[i - 1]) or biases[i].shape != (shape[i], 1):
                msg = "loaded parameters do not fit the network"
                msg += f": expected parameters suitable for network shape {self.shape}"
                raise ValueError(msg)
        self.weights = weights
        self.biases = biases

    def __repr__(self):
        return f"Network{self.shape}"
