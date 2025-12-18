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
            activation = "relu"
            
            # --- MODIFICATION : Activation Linear pour la dernière couche ---
            # Permet au Softmax (dans forward) de travailler sur des valeurs brutes
            if i == self.layerCount - 1:
                activation = "linear"

            # --- MODIFICATION : Dropout augmenté à 0.2 (20%) ---
            # Force le réseau à généraliser au lieu d'apprendre par cœur
            layer = Layer(l1, l2, activation, dropout_rate=0.2)

            if weights is not None and i-1 < len(weights):
                layer.weights = weights[i-1]
                layer.biases = biases[i-1]
            else:
                limit = np.sqrt(2.0 / l1)
                layer.weights = np.random.randn(l1, l2) * limit
                layer.biases = np.zeros((1, l2))

            self.layers.append(layer)

    # Mini batch and Shuffling
    def train(self, learningRate, saveFile, batch_size=32, X_val=None, Y_val=None):
        """Train the network using the input and output data.
        Args:
            learningRate (float): Learning rate for gradient descent.
            saveFile (str): Path to save the trained network.
            batch_size (int): Size of mini-batches.
            X_val (numpy.ndarray): Validation input data.
            Y_val (numpy.ndarray): Validation target data.
        """
        num_samples = len(self.matrix_input)
        print(f"Starting training on {num_samples} samples...")

        for epoch in range(EPOCH):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = self.matrix_input[indices]
            Y_shuffled = self.matrix_output[indices]

            total_loss = 0.0
            total_correct = 0 # Compteur pour l'accuracy d'entraînement

            for i in range(0, num_samples, batch_size):
                input_data = X_shuffled[i:i + batch_size]
                expected_output = Y_shuffled[i:i + batch_size]
                current_batch_size = len(input_data)

                # Propagation avant
                predicted_output = self.forward(input_data, training=True)

                # Calcul Loss
                epsilon = 1e-15
                predicted_clipped = np.clip(predicted_output, epsilon, 1 - epsilon)
                loss = -np.sum(expected_output * np.log(predicted_clipped)) / current_batch_size
                total_loss += loss * current_batch_size

                train_preds = np.argmax(predicted_output, axis=1)
                train_labels = np.argmax(expected_output, axis=1)
                total_correct += np.sum(train_preds == train_labels)

                # Gradient (P - Y)
                gradient = (predicted_output - expected_output)

                # Rétropropagation
                self.backward(gradient, learningRate)

            # Moyennes de l'époque
            avg_loss = total_loss / num_samples
            train_acc = total_correct / num_samples

            val_msg = ""
            if X_val is not None and Y_val is not None:
                val_output = self.forward(X_val, training=False)
                
                val_preds = np.argmax(val_output, axis=1)
                val_truth = np.argmax(Y_val, axis=1)
                val_accuracy = np.mean(val_preds == val_truth)
                
                val_msg = f" - Val Acc: {val_accuracy:.2%}"

                # Si on dépasse 96% de réussite sur des données inconnues, on arrête !
                if val_accuracy > 0.96:
                    print(f"Epoch {epoch + 1}/{EPOCH} - Loss: {avg_loss:.6f} - Train Acc: {train_acc:.2%}{val_msg}")
                    print("🚀 Objectif atteint (Validation > 96%) ! Arrêt anticipé pour éviter le surentraînement.")
                    break

            if (epoch + 1) % 20 == 0:
                learningRate *= 0.9
                print(f"Learning rate reduced to {learningRate:.5f}")

            print(f"Epoch {epoch + 1}/{EPOCH} - Loss: {avg_loss:.6f} - Train Acc: {train_acc:.2%}{val_msg}")

        self.saveTrainedNetwork(saveFile)
        print(f"Network saved to {saveFile}")

    def predict(self):
        """Predict the class for a given input with probabilities."""
        output = self.forward(self.matrix_input, training=False)
        predictions = np.argmax(output, axis=1)
        mapping = {0: "Nothing", 1: "Check", 2: "Checkmate"}

        print(f"{'PREDICTION':<12} | {'CONFIANCE':<10} | {'DÉTAIL [Nothing, Check, Mate]'}")
        print("-" * 60)

        for i, p in enumerate(predictions):
            label = mapping.get(p, "Error")
            confidence = output[i][p]
            # On formate les 3 probas pour voir les hésitations
            probas_str = f"[{output[i][0]:.3f}, {output[i][1]:.3f}, {output[i][2]:.3f}]"

            print(f"{label:<12} | {confidence:.1%}      | {probas_str}")

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