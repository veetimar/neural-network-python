import unittest

import numpy as np

from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3, 1)
        self.nn = Network(self.shape)
        self.xor = [([0, 0], [0]),([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

    def test_constructor_raises_error(self):
        self.assertRaises(ValueError, Network, (0, 1))
        self.assertRaises(ValueError, Network, (1, 0))
        self.assertRaises(ValueError, Network, (1,))

    def test_constructor_sets_attributes_shape_right(self):
        self.assertEqual(len(self.nn.values), len(self.shape))
        self.assertEqual(len(self.nn.activations), len(self.shape))
        self.assertEqual(len(self.nn.gradients), len(self.shape))
        self.assertEqual(len(self.nn.weights), len(self.shape))
        self.assertEqual(len(self.nn.biases), len(self.shape))
        self.assertEqual(len(self.nn.delta_weights), len(self.shape))
        self.assertEqual(len(self.nn.delta_biases), len(self.shape))
        for i in range(1, len(self.shape)):
            self.assertEqual(self.nn.weights[i].shape, (self.shape[i], self.shape[i - 1]))
            self.assertEqual(self.nn.biases[i].shape, (self.shape[i], 1))
            self.assertEqual(self.nn.delta_weights[i].shape, self.nn.weights[i].shape)
            self.assertEqual(self.nn.delta_biases[i].shape, self.nn.biases[i].shape)

    def test_glorot_returns_right_shape(self):
        weights = self.nn.glorot(3, 2, 1)
        self.assertEqual(weights.shape, (2, 3))

    def test_he_returns_right_shape(self):
        weights = self.nn.he(3, 2)
        self.assertEqual(weights.shape, (2, 3))

    def test_sigmoid_returns_right_shape(self):
        in_1 = np.zeros((2))
        in_2 = np.zeros((2, 1))
        in_3 = np.zeros((1, 2))
        out_1 = self.nn.sigmoid(in_1)
        out_2 = self.nn.sigmoid(in_2)
        out_3 = self.nn.sigmoid(in_3)
        self.assertEqual(out_1.shape, in_1.shape)
        self.assertEqual(out_2.shape, in_2.shape)
        self.assertEqual(out_3.shape, in_3.shape)

    def test_sigmoid_returns_right_value(self):
        inputs = np.array([-1, 0, 1])
        outputs = self.nn.sigmoid(inputs)
        expected = np.array([0.27, 0.5, 0.73])
        self.assertTrue(np.allclose(outputs, expected, atol=0.01))

    def test_sigmoid_derivative_returns_right_shape(self):
        in_1 = np.zeros((2))
        in_2 = np.zeros((2, 1))
        in_3 = np.zeros((1, 2))
        out_1 = self.nn.sigmoid_derivative(in_1)
        out_2 = self.nn.sigmoid_derivative(in_2)
        out_3 = self.nn.sigmoid_derivative(in_3)
        self.assertEqual(out_1.shape, in_1.shape)
        self.assertEqual(out_2.shape, in_2.shape)
        self.assertEqual(out_3.shape, in_3.shape)

    def test_sigmoid_derivative_returns_right_value(self):
        inputs = np.array([-1, 0, 1])
        outputs = self.nn.sigmoid_derivative(inputs)
        expected = np.array([0.2, 0.25, 0.2])
        self.assertTrue(np.allclose(outputs, expected, atol=0.01))

    def test_elu_returns_right_shape(self):
        in_1 = np.zeros((2))
        in_2 = np.zeros((2, 1))
        in_3 = np.zeros((1, 2))
        out_1 = self.nn.elu(in_1)
        out_2 = self.nn.elu(in_2)
        out_3 = self.nn.elu(in_3)
        self.assertEqual(out_1.shape, in_1.shape)
        self.assertEqual(out_2.shape, in_2.shape)
        self.assertEqual(out_3.shape, in_3.shape)

    def test_elu_returns_right_value(self):
        inputs = np.array([-1, 0, 1])
        outputs = self.nn.elu(inputs)
        expected = np.array([-0.63, 0, 1])
        self.assertTrue(np.allclose(outputs, expected, atol=0.01))

    def test_elu_derivative_returns_right_shape(self):
        in_1 = np.zeros((2))
        in_2 = np.zeros((2, 1))
        in_3 = np.zeros((1, 2))
        out_1 = self.nn.elu_derivative(in_1)
        out_2 = self.nn.elu_derivative(in_2)
        out_3 = self.nn.elu_derivative(in_3)
        self.assertEqual(out_1.shape, in_1.shape)
        self.assertEqual(out_2.shape, in_2.shape)
        self.assertEqual(out_3.shape, in_3.shape)

    def test_elu_derivative_returns_right_value(self):
        inputs = np.array([-1, 0, 1])
        outputs = self.nn.elu_derivative(inputs)
        expected = np.array([0.37, 1, 1])
        self.assertTrue(np.allclose(outputs, expected, atol=0.01))

    def test_mean_squared_returns_right_shape(self):
        ex_1 = np.zeros((2))
        ex_2 = np.zeros((2, 1))
        ex_3 = np.zeros((1, 2))
        err_1 = self.nn.mean_squared(ex_1, ex_1)
        self.assertRaises(TypeError, iter, err_1)
        err_2 = self.nn.mean_squared(ex_2, ex_2)
        self.assertRaises(TypeError, iter, err_2)
        err_3 = self.nn.mean_squared(ex_3, ex_3)
        self.assertRaises(TypeError, iter, err_3)

    def test_mean_squared_returns_right_value(self):
        error = self.nn.mean_squared([1, 3], [2, 1])
        real = 2.5
        self.assertEqual(error, real)

    def test_mean_squared_gradient_returns_right_shape(self):
        ex_1 = np.zeros(2)
        ex_2 = np.zeros((2, 1))
        ex_3 = np.zeros((1, 2))
        grad_1 = self.nn.mean_squared_gradient(ex_1, ex_1)
        self.assertEqual(ex_1.shape, grad_1.shape)
        grad_2 = self.nn.mean_squared_gradient(ex_2, ex_2)
        self.assertEqual(ex_2.shape, grad_2.shape)
        grad_3 = self.nn.mean_squared_gradient(ex_3, ex_3)
        self.assertEqual(ex_3.shape, grad_3.shape)

    def test_mean_squared_gradient_returns_right_value(self):
        outputs = np.array([1, 3])
        expected = np.array([2, 1])
        gradient = self.nn.mean_squared_gradient(outputs, expected)
        real = np.array([-1, 2])
        self.assertTrue((gradient == real).all())

    def test_forward_returns_right_shape(self):
        outputs = self.nn.forward(self.xor[0][0])
        self.assertEqual(outputs.shape, (self.shape[-1], 1))

    def test_backward_returns_right_shape(self):
        error = self.nn.backward(self.xor[0][0], self.xor[0][1])
        self.assertRaises(TypeError, iter, error)

    def test_train_returns_right_shape(self):
        epochs = 10
        errors = self.nn.train(self.xor, epochs, batch_size=3)
        self.assertEqual(len(errors), epochs)

    def test_train_raises_error(self):
        data = self.xor
        self.assertRaises(ValueError, self.nn.train, data, 0)
        self.assertRaises(ValueError, self.nn.train, data, 1, 0)
        data[0] = ([0, 0], [1, 1], [0, 0])
        self.assertRaises(ValueError, self.nn.train, data, 1)

    def test_save_and_load(self):
        old_weights = self.nn.weights
        old_biases = self.nn.biases
        self.nn.save(".")
        new_network = Network(self.shape)
        new_network.load(".")
        new_weights = new_network.weights
        new_biases = new_network.biases
        self.assertEqual(len(old_weights), len(new_weights))
        self.assertEqual(len(old_biases), len(new_biases))
        for i in range(1, len(self.shape)):
            self.assertTrue((old_weights[i] == new_weights[i]).all())
            self.assertTrue((old_biases[i] == new_biases[i]).all())

    def test_load_raises_error(self):
        self.nn.save()
        shape = list(self.shape)
        self.assertRaises(ValueError, Network(shape[:-1]).load)
        shape[0] += 1
        self.assertRaises(ValueError, Network(shape).load)

    def test_repr(self):
        self.assertEqual(repr(self.nn), f"Network{self.shape}")

    def test_network_overfits_xor(self):
        self.nn.train(self.xor, 5000, learning_rate=10)
        for data in self.xor:
            outputs = self.nn.forward(data[0]).reshape(-1)
            self.assertTrue(np.allclose(data[1], outputs, atol=0.01))

    def test_every_weight_changes(self):
        old_weights = [array.copy() for array in self.nn.weights[1:]]
        old_biases = [array.copy() for array in self.nn.biases[1:]]
        self.nn.train(self.xor, 100)
        new_weights = [array.copy() for array in self.nn.weights[1:]]
        new_biases = [array.copy() for array in self.nn.biases[1:]]
        for i in range(len(self.shape) - 1):
            self.assertTrue((old_weights[i] != new_weights[i]).all())
            self.assertTrue((old_biases[i] != new_biases[i]).all())

    def test_backpropagation_does_not_touch_input_layer(self):
        self.nn.train(self.xor, 1)
        self.assertIsNone(self.nn.gradients[0])
        self.assertIsNone(self.nn.weights[0])
        self.assertIsNone(self.nn.biases[0])
        self.assertIsNone(self.nn.delta_weights[0])
        self.assertIsNone(self.nn.delta_biases[0])
