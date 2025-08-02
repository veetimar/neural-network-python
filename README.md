## About
A fully connected neural network, ELU activation in hidden layers and sigmoid in output layer.
Comes with a simple user interface for training and testing the neural net with MNIST hand-written digits but can be used for practically any task since the size of the neural net and batch size can be freely choosed. Is able to achieve over 97% accuracy in the MNIST dataset with just a few minutes of training.
## How to use
Clone the repository and install the required dependencies with:

```console
poetry install --without dev
```

If you wish to use the neural network for something else than the included UI you can run

```console
poetry install --without dev,run
```

instead.

To run the digits recognition UI:

```console
poetry run python src/digits.py
```

Docstrings in the code should give a clear enough view on how the network class can be used in you own code. Mainly look at "forward" and "train" methods.
