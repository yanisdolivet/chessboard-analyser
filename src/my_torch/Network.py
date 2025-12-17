##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Network
##

import struct
import sys
import numpy as np
from src.my_torch.Layer import Layer

MAGIC_NUMBER = 0x48435254
ERROR_CODE = 84
EPOCH = 100

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

            if weights is not None and i-1 < len(weights):
                layer.weights = weights[i-1]
                layer.biases = biases[i-1]
            else:
                limit = np.sqrt(2.0 / l1)
                layer.weights = np.random.randn(l1, l2) * limit
                layer.biases = np.zeros((1, l2))

            self.layers.append(layer)

    # Mini batch and Shuffling
    def train(self, learningRate, saveFile, batch_size=32):
        """Train the network using the input and output data.
        Args:
            learningRate (float): Learning rate for gradient descent.
            saveFile (str): Path to save the trained network.
        """
        num_samples = len(self.matrix_input)

        for epoch in range(EPOCH):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = self.matrix_input[indices]
            Y_shuffled = self.matrix_output[indices]

            total_loss = 0.0
            # Boucle sur les mini-batchs pour plus de rapidité
            for i in range(0, num_samples, batch_size):
                # Utilisation des données mélangées
                input_data = X_shuffled[i:i + batch_size]
                expected_output = Y_shuffled[i:i + batch_size]

                # Propagation avant sur le batch complet
                predicted_output = self.forward(input_data)

                # Cross-Entropy
                epsilon = 1e-15
                predicted_clipped = np.clip(predicted_output, epsilon, 1 - epsilon)
                # Calcul de la perte moyenne sur le batch (stable)
                loss = -np.sum(expected_output * np.log(predicted_clipped))
                total_loss += loss

                gradient = (predicted_output - expected_output) / len(input_data)

                # Rétropropagation
                self.backward(gradient, learningRate)

            if (epoch + 1) % 10 == 0:
                learningRate *= 0.5
                print(f"Learning rate reduced to {learningRate:.5f}")
                avg_loss = total_loss / num_samples
                print(f"Epoch {epoch + 1}/{EPOCH} - Loss: {avg_loss:.6f}")

        self.saveTrainedNetwork(saveFile)
        print(f"Network saved to {saveFile} after epoch {epoch + 1}")

    def predict(self):
        """Predict the class for a given input.
        Args:
            input: Input data to predict.
        Returns:
            int: Predicted class index.
        """
        output = self.forward(self.matrix_input, training=False)

        print(f"output = {output}")
        predictions = np.argmax(output, axis=1)

        mapping = {0: "Nothing", 1: "Check", 2: "Checkmate"}
        for p in predictions:
            print(mapping.get(p, "Error"))

    def forward(self, input, training=True) -> np.array:
        current = input
        for i in range(len(self.layers)):
            current = self.layers[i].forward(current, training)

            if i == len(self.layers) - 1:
                shift_current = current - np.max(current, axis=1, keepdims=True)
                exps = np.exp(shift_current)
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
            with open(filePath, 'wb') as f:
                # Write header (magic number and layer count)
                f.write(struct.pack('II', MAGIC_NUMBER, len(self.layerSize)))

                # Write layer sizes (topology)
                f.write(struct.pack(f'{len(self.layerSize)}I', *self.layerSize))

                # Write all weights
                for layer in self.layers:
                    w_flat = layer.weights.astype(np.float32).flatten()
                    f.write(struct.pack(f'{len(w_flat)}f', *w_flat))

                # Write all biases
                for layer in self.layers:
                    b_flat = layer.biases.astype(np.float32).flatten()
                    f.write(struct.pack(f'{len(b_flat)}f', *b_flat))

            print(f"Saved trained network to {filePath}")
        except IOError as e:
            print(f"IOError while saving the network: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)