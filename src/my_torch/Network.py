##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Network
##

import struct
import sys
from src.my_torch.Layer import Layer
import numpy as np

MAGIC_NUMBER = 0x48435254
ERROR_CODE = 84


class Network:

    def __init__(self, layerSize, matrixInput, matrixOutput):
        self.layerCount = len(layerSize)  # number of layer
        self.layerSize = (
            layerSize  # list that contains the nb of parameter for each layer
        )
        self.matrix_input = matrixInput
        self.matrix_output = matrixOutput
        self.layers = []

    def createLayer(self, weights, biases):
        for i in range(1, self.layerCount):
            l1 = self.layerSize[i - 1]
            l2 = self.layerSize[i]

            layer = Layer(l1, l2)

            layer.weights = weights[i - 1]
            layer.biases = biases[i - 1]
            self.layers.append(layer)

    def train(self, learningRate, saveFile):
        """Train the network using the input and output data.
        Args:
            learningRate (float): Learning rate for gradient descent.
            saveFile (str): Path to save the trained network.
        """
        for epoch in range(50):
            total_loss = 0.0
            for i in range(len(self.matrix_input)):
                input_data = self.matrix_input[i]

                expected_output = self.matrix_output[i]
                predicted_output = self.forward(input_data)

                # Cross-Entropy
                epsilon = 1e-15
                predicted_clipped = np.clip(predicted_output, epsilon, 1 - epsilon)
                loss = -np.sum(expected_output * np.log(predicted_clipped))
                total_loss += loss

                gradient = predicted_output - expected_output

                self.backward(gradient, learningRate)

            avg_loss = total_loss / len(self.matrix_input)
            print(f"Epoch {epoch + 1}/50 - Loss: {avg_loss:.6f}")
        self.saveTrainedNetwork(saveFile)
        print(f"Network saved to {saveFile} after epoch {epoch + 1}")

    def predict(self):
        """Predict the class for a given input.
        Args:
            input: Input data to predict.
        Returns:
            int: Predicted class index.
        """
        output = None
        for i in range(len(self.matrix_input)):
            output = self.forward(self.matrix_input[i])
        prediction = np.argmax(output)
        print(f"Output probabilities: {output}")
        if prediction == 0:
            print("Nothing")
        elif prediction == 1:
            print("Check")
        elif prediction == 2:
            print("Checkmate")

    def forward(self, input_data) -> np.array:
        current = input_data
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            if i == len(self.layers) - 1:
                exps = np.exp(current - np.max(current, axis=1, keepdims=True))
                current = exps / np.sum(exps, axis=1, keepdims=True)
        return current

    def backward(self, gradient, learning_rate=0.01):
        """Perform backward pass through all layers.
        Args:
            gradient (numpy.ndarray): Initial gradient from loss function.
            learning_rate (float): Learning rate for updates.

        Returns:
            numpy.ndarray: Gradient propagated to input layer.
        """
        current_gradient = gradient

        for i in range(len(self.layers) - 1, -1, -1):
            current_gradient = self.layers[i].backward(current_gradient, learning_rate)

        return current_gradient

    def saveTrainedNetwork(self, filePath):
        try:
            with open(filePath, "wb") as f:
                f.write(struct.pack("II", MAGIC_NUMBER, len(self.layerSize)))

                f.write(struct.pack(f"{len(self.layerSize)}I", *self.layerSize))

                for w in self.layers:
                    w_flat = w.weights.flatten()
                    f.write(struct.pack(f"{len(w_flat)}f", *w_flat))
                for b in self.layers:
                    b_flat = b.biases.flatten()
                    f.write(struct.pack(f"{len(b_flat)}f", *b_flat))
            print(f"Saved trained network to {filePath}")
        except IOError as e:
            print(f"IOError while saving the network: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)
