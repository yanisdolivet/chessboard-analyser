##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Network (CORRECTED)
##

import struct
import sys
import copy
import numpy as np
from src.my_torch.Layer import Layer
from src.data_analysis.data_analysis import DataAnalysis

MAGIC_NUMBER = 0x48435254
ERROR_CODE = 84


class Network:

    def __init__(self, layerSize, matrixInput, matrixOutput, model_spec):
        self.layerCount = len(layerSize)
        self.layerSize = layerSize
        self.matrix_input = matrixInput
        self.matrix_output = matrixOutput
        self.layers = []
        self.model_spec = model_spec
        self.data_analysis = DataAnalysis(
            modelspec=model_spec, epochs=model_spec.epochs
        )

    def createLayer(self, weights, biases):
        for i in range(1, self.layerCount):
            l1 = self.layerSize[i - 1]
            l2 = self.layerSize[i]

            if i == self.layerCount - 1:
                activation = "linear"
            else:
                activation = (
                    self.model_spec.activation[i]
                    if i < len(self.model_spec.activation)
                    else "relu"
                )

            layer = Layer(l1, l2, activation, dropout_rate=self.model_spec.dropout_rate)

            if weights is not None and i - 1 < len(weights):
                layer.weights = weights[i - 1]
                layer.biases = biases[i - 1]
            else:
                # Use initialization from model spec
                if (
                    self.model_spec.initialization == "he_normal"
                    or self.model_spec.initialization == "he_mixed_xavier"
                ):
                    limit = np.sqrt(2.0 / l1)
                elif self.model_spec.initialization == "xavier":
                    limit = np.sqrt(1.0 / l1)
                else:
                    limit = 0.01
                layer.weights = np.random.randn(l1, l2) * limit
                layer.biases = np.zeros((1, l2))

            self.layers.append(layer)

    def train(
        self,
        learningRate,
        saveFile,
        batch_size=None,
        X_val=None,
        Y_val=None,
        data_analysis=None,
    ):
        """Entraînement avec Validation et Sauvegarde du Meilleur Modèle."""
        num_samples = len(self.matrix_input)
        print(f"Starting training on {num_samples} samples...")

        if batch_size is None:
            batch_size = self.model_spec.batch_size

        if learningRate is None:
            learningRate = self.model_spec.learning_rate

        self.data_analysis = data_analysis

        best_val_acc = 0.0
        best_weights = None
        best_biases = None

        for epoch in range(self.model_spec.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = self.matrix_input[indices]
            Y_shuffled = self.matrix_output[indices]

            total_loss = 0.0
            total_correct = 0

            for i in range(0, num_samples, batch_size):
                input_data = X_shuffled[i : i + batch_size]
                expected_output = Y_shuffled[i : i + batch_size]
                current_batch_size = len(input_data)

                # Forward
                predicted_output = self.forward(input_data, training=True)

                # Save predictions for analysis (if data_analysis is available)
                if self.data_analysis:
                    self.data_analysis.save_predictions(
                        np.argmax(predicted_output, axis=1).tolist(),
                        np.argmax(expected_output, axis=1).tolist(),
                    )

                # Loss (Cross-Entropy)
                epsilon = 1e-15
                predicted_clipped = np.clip(predicted_output, epsilon, 1 - epsilon)
                loss = (
                    -np.sum(expected_output * np.log(predicted_clipped))
                    / current_batch_size
                )

                # L2 regulatization
                w = 0
                for i in range(len(self.layers)):
                    w += np.sum(np.square(self.layers[i].weights))
                scale_w = (self.model_spec.lreg / 2) * w
                total_loss += (loss + scale_w) * current_batch_size

                # Accuracy Train
                train_preds = np.argmax(predicted_output, axis=1)
                train_labels = np.argmax(expected_output, axis=1)
                total_correct += np.sum(train_preds == train_labels)

                # Backward
                gradient = predicted_output - expected_output
                self.backward(gradient, learningRate, self.model_spec.lreg)

            # Metrics
            avg_loss = total_loss / num_samples
            train_acc = total_correct / num_samples

            val_msg = ""
            if X_val is not None and Y_val is not None:
                val_output = self.forward(X_val, training=False)
                val_preds = np.argmax(val_output, axis=1)
                val_truth = np.argmax(Y_val, axis=1)
                val_acc = np.mean(val_preds == val_truth)
                val_loss = -np.sum(
                    Y_val * np.log(np.clip(val_output, 1e-15, 1 - 1e-15))
                ) / len(X_val)
                val_msg = f" - Val Acc: {val_acc:.2%}"

                # Sauvegarde du meilleur état (à garder ??)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = [copy.deepcopy(l.weights) for l in self.layers]
                    best_biases = [copy.deepcopy(l.biases) for l in self.layers]

            print(
                f"Epoch {epoch + 1}/{self.model_spec.epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2%}{val_msg}"
            )

            if self.data_analysis:
                self.data_analysis.save_loss(
                    avg_loss, val_loss if X_val is not None else 0.0
                )
                self.data_analysis.save_metrics(
                    train_acc, val_acc if X_val is not None else 0.0
                )

            # Learning rate decay doux
            if (epoch + 1) % 20 == 0:
                learningRate *= 0.9

        # Restauration du meilleur modèle
        if best_weights is not None:
            print(f"Restoring best model (Val Acc: {best_val_acc:.2%})")
            for i, layer in enumerate(self.layers):
                layer.weights = best_weights[i]
                layer.biases = best_biases[i]

        self.saveTrainedNetwork(saveFile)

        if self.data_analysis:
            self.data_analysis.export()

    def predict(self):
        """Affiche les prédictions avec les probabilités détaillées."""
        output = self.forward(self.matrix_input, training=False)
        predictions = np.argmax(output, axis=1)
        mapping = {0: "Nothing", 1: "Check", 2: "Checkmate"}

        for i, p in enumerate(predictions):
            label = mapping.get(p, "Error")
            print(label)

    def forward(self, input, training=True) -> np.array:
        current = input
        for i in range(len(self.layers)):
            current = self.layers[i].forward(current, training)

            if i == len(self.layers) - 1:
                shift_current = current - np.max(current, axis=1, keepdims=True)
                exps = np.exp(shift_current)
                current = exps / np.sum(exps, axis=1, keepdims=True)
        return current

    def backward(self, gradient, learning_rate, lambda_reg):
        """Perform backward pass through all layers with L2 regularization.

        Args:
            gradient (numpy.ndarray): Initial gradient from loss function.
            learning_rate (float): Learning rate for updates.
            lambda_reg (float): L2 regularization strength.

        Returns:
            numpy.ndarray: Gradient propagated to input layer.
        """
        current_gradient = gradient
        for i in range(len(self.layers) - 1, -1, -1):
            current_gradient = self.layers[i].backward(
                current_gradient, learning_rate, lambda_reg
            )
        return current_gradient

    def _encode_string(self, s):
        """Encode string as length-prefixed bytes."""
        encoded = s.encode("utf-8")
        return struct.pack("I", len(encoded)) + encoded

    def saveTrainedNetwork(self, filePath):
        try:
            VERSION = 2
            with open(filePath, "wb") as f:
                f.write(struct.pack("III", MAGIC_NUMBER, VERSION, len(self.layerSize)))
                f.write(struct.pack(f"{len(self.layerSize)}I", *self.layerSize))

                for i in range(len(self.model_spec.type)):
                    f.write(self._encode_string(self.model_spec.type[i]))
                    f.write(self._encode_string(self.model_spec.activation[i]))

                f.write(struct.pack("f", self.model_spec.learning_rate))
                f.write(self._encode_string(self.model_spec.initialization))

                f.write(struct.pack("I", self.model_spec.batch_size))
                f.write(struct.pack("I", self.model_spec.epochs))
                f.write(struct.pack("f", self.model_spec.lreg))
                f.write(struct.pack("f", self.model_spec.dropout_rate))
                f.write(self._encode_string(self.model_spec.loss_function))

                for layer in self.layers:
                    w_flat = layer.weights.astype(np.float32).flatten()
                    f.write(struct.pack(f"{len(w_flat)}f", *w_flat))
                for layer in self.layers:
                    b_flat = layer.biases.astype(np.float32).flatten()
                    f.write(struct.pack(f"{len(b_flat)}f", *b_flat))

            print(f"Saved trained network to {filePath}")
        except IOError as e:
            print(f"IOError: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)
