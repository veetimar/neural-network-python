import unittest

from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.size = (2, 3, 1)
        self.network = Network(*self.size)

    def test_constructor_raises_error(self):
        self.assertRaises(ValueError, Network, 0, 1)
        self.assertRaises(ValueError, Network, 1, 0)
        self.assertRaises(ValueError, Network, 1)

    def test_constructor_sets_attributes(self):
        self.assertEqual(self.size, self.network.size)
        for i, length in enumerate(self.size):
            self.assertEqual(len(self.network.values[i]), length)
            if i == 0:
                self.assertIsNone(self.network.weights[i])
                self.assertIsNone(self.network.biases[i])
            else:
                self.assertEqual(len(self.network.weights[i]), length)
                self.assertEqual(len(self.network.biases[i]), length)

    def test_repr(self):
        self.assertEqual(repr(self.network), f"Network{self.size}")
