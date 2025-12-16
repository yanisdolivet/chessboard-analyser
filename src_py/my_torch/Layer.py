##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Layer
##

import numpy as np

class Layer:

    def __init__(self, inputSize, outputSize):
        """Initialize a neural network layer.

        Args:
            inputSize (int): Number of inputs to this layer.
            outputSize (int): Number of outputs from this layer.
        """
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.weights = None
        self.biases = None

        # Set activation function
        self.activation_type = "relu"
        if self.activation_type == "relu":
            self.act_func = lambda x: np.maximum(0, x)
        else:
            self.act_func = lambda x: x

        self.cache_input = None
        self.cache_z = None
        self.cache_a = None

    def forward(self, input):
        self.cache_input = input

        z = np.add(np.dot(input, self.weights), self.biases)
        self.cache_z = z

        a = (self.act_func)(z)
        self.cache_a
        return a