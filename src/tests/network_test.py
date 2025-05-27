import unittest

from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.size = (3, 4, 2)
        self.network = Network(self.size)
        self.zeros = [0 for _ in range(self.size[0])]

    def test_constructor_raises_error(self):
        self.assertRaises(ValueError, Network, (0, 1))
        self.assertRaises(ValueError, Network, (1, 0))
        self.assertRaises(ValueError, Network, (1,))

    def test_constructor_sets_attributes_shape_right(self):
        self.assertEqual(len(self.network.values), len(self.size))
        self.assertEqual(len(self.network.weights), len(self.size))
        self.assertEqual(len(self.network.biases), len(self.size))
        for i in range(1, len(self.size)):
            self.assertEqual(self.network.weights[i].shape, (self.size[i], self.size[i - 1]))
            self.assertEqual(self.network.biases[i].shape, (self.size[i], 1))

    def test_forward_returns_right_shape(self):
        self.assertEqual(self.network.forward(self.zeros).shape, (self.size[-1], 1))

    def test_repr(self):
        self.assertEqual(repr(self.network), f"Network{self.size}")
