import unittest

from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 4, 2)
        self.network = Network(self.shape)
        self.zeros = [0 for _ in range(self.shape[0])]

    def test_constructor_raises_error(self):
        self.assertRaises(ValueError, Network, (0, 1))
        self.assertRaises(ValueError, Network, (1, 0))
        self.assertRaises(ValueError, Network, (1,))

    def test_constructor_sets_attributes_shape_right(self):
        self.assertEqual(len(self.network.values), len(self.shape))
        self.assertEqual(len(self.network.weights), len(self.shape))
        self.assertEqual(len(self.network.biases), len(self.shape))
        for i in range(1, len(self.shape)):
            self.assertEqual(self.network.weights[i].shape, (self.shape[i], self.shape[i - 1]))
            self.assertEqual(self.network.biases[i].shape, (self.shape[i], 1))

    def test_forward_returns_right_shape(self):
        self.assertEqual(self.network.forward(self.zeros).shape, (self.shape[-1], 1))

    def test_save_and_load(self):
        old_weights = self.network.weights
        old_biases = self.network.biases
        self.network.save()
        new_network = Network(self.shape)
        new_network.load()
        new_weights = new_network.weights
        new_biases = new_network.biases
        self.assertEqual(len(old_weights), len(new_weights))
        self.assertEqual(len(old_biases), len(new_biases))
        for i in range(1, len(self.shape)):
            self.assertTrue((old_weights[i] == new_weights[i]).all())
            self.assertTrue((old_biases[i] == new_biases[i]).all())

    def test_load_raises_error(self):
        self.network.save()
        shape = list(self.shape)
        self.assertRaises(ValueError, Network(shape[:-1]).load)
        shape[0] += 1
        self.assertRaises(ValueError, Network(shape).load)

    def test_repr(self):
        self.assertEqual(repr(self.network), f"Network{self.shape}")
