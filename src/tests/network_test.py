import unittest

from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.size = (3, 4, 2)
        self.network = Network(list(self.size))

    def test_constructor_raises_error(self):
        self.assertRaises(ValueError, Network, (0, 1))
        self.assertRaises(ValueError, Network, (1, 0))
        self.assertRaises(ValueError, Network, (1,))

    def test_constructor_sets_attributes_shape_right(self):
        self.assertEqual(len(self.network.values), len(self.size))
        self.assertEqual(len(self.network.weights), len(self.size))
        self.assertEqual(len(self.network.biases), len(self.size))
        for i in range(1, len(self.size)):
            self.assertEqual(len(self.network.weights[i]), self.size[i])
            self.assertEqual(len(self.network.weights[i][0]), self.size[i - 1])
            self.assertEqual(len(self.network.biases[i]), self.size[i])

    def test_forward_returns_right_shape(self):
        self.assertEqual(len(self.network.forward([0 for _ in range(self.size[0])])), self.size[-1])

    def test_repr(self):
        self.assertEqual(repr(self.network), f"Network{self.size}")
