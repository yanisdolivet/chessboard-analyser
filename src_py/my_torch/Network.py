##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Network
##

from src_py.my_torch.Layer import Layer
import numpy as np

class Network:

    def __init__(self, layerSize, matrixInput, matrixOutput):
        self.layerCount = len(layerSize) # number of layer
        self.layerSize = layerSize # list that contains the nb of parameter for each layer
        self.matrix_input = matrixInput
        self.matrix_output = matrixOutput
        self.layers = []

    def createLayer(self, weights, biases):
        for i in range(1, self.layerCount):
            l1 = self.layerSize[i-1]
            l2 = self.layerSize[i]

            layer = Layer(l1, l2)

            layer.weights = weights[i-1]
            layer.biases = biases[i-1]
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

    def predict(self):
        for input in self.matrix_input:
            output = self.forward(input)
        pass

    def forward(self, input) -> np.array:
        current = input
        for i in range(len(self.layers)):
            current = self.layers[i].forward(current)
        return current

    def backward(self, gradient, learning_rate=0.01):
        pass

