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
        """Perform forward pass through the layer.

        Args:
            input (numpy.ndarray): Input data, can be 1D or 2D.

        Returns:
            numpy.ndarray: Activated output.
        """
        if input.ndim == 1:
            input = input.reshape(1, -1)

        self.cache_input = input

        z = np.add(np.dot(input, self.weights), self.biases)
        self.cache_z = z

        a = (self.act_func)(z)
        self.cache_a = a
        return a

    def activate_derivative(self, z):
        """Compute the derivative of the activation function at z.

        Args:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the activation function.
        """
        if self.activation_type == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation_type == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)

    def backward(self, gradient, learning_rate):
        """Perform backward pass (backpropagation).

        Args:
            gradient (numpy.ndarray): Gradient from the next layer.
            learning_rate (float): Learning rate for weight updates.

        Returns:
            numpy.ndarray: Gradient to pass to the previous layer.
        """
        # how much to change my answer
        derivative = self.activate_derivative(self.cache_z)
        dZ = gradient * derivative

        # Calculate how much to change weights and biases
        input_T = np.transpose(self.cache_input)
        dW = np.dot(input_T, dZ)
        dB = np.sum(dZ, axis=0, keepdims=True)

        # Pass the error to the previous layer
        weight_T = np.transpose(self.weights)
        dX_prev = np.dot(dZ, weight_T)

        # Update weights and biases
        step_W = dW * learning_rate
        step_B = dB * learning_rate

        self.weights = self.weights - step_W
        self.biases = self.biases - step_B

        return dX_prev