[![CI](https://github.com/veetimar/neural-network-python/actions/workflows/main.yml/badge.svg)](https://github.com/veetimar/neural-network-python/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/veetimar/neural-network-python/graph/badge.svg?token=B3NIJQEDA6)](https://codecov.io/gh/veetimar/neural-network-python)

## About

A fully connected neural network, ELU activation in hidden layers and sigmoid in output layer.
Comes with a simple user interface for training and testing the neural net with MNIST hand-written digits but can be used for practically any task since the size of the neural net and batch size can be freely choosed. Is able to achieve over 97% accuracy in the MNIST dataset with just a few minutes of training.

## How to use

### The easy way (Docker installed)

```console
docker run -itv parameters:/workdir/parameters veetimar/neural-network
```

Note that matplotlib seems not to work inside a container so plotting error after training and showing images classified wrong/right do not work when using Docker.

### The hard way (Poetry and Python installed)

Clone the repository

```console
git clone https://github.com/veetimar/neural-network-python
```

Install the required dependencies with

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
