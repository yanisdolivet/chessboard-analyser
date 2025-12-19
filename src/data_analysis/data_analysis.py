##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## data_analysis
##

import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "docs/benchmarks/"

class DataAnalysis:
    def __init__(self, epochs=100):
        self.training_loss = []
        self.validation_loss = []

        self.training_metrics = []
        self.validation_metrics = []
        self.epochs = epochs

        self.prediction = []
        self.expected = []

    def save_loss(self, training_loss, validation_loss):
        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)

    def save_metrics(self, training_metric, validation_metric):
        self.training_metrics.append(training_metric)
        self.validation_metrics.append(validation_metric)

    def save_predictions(self, preds, truths):
        self.prediction.extend(preds)
        self.expected.extend(truths)

    def build_confusion_matrix(self):
        num_classes = 3
        confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

        for true, pred in zip(self.expected, self.prediction):
            confusion_matrix[true][pred] += 1

        return confusion_matrix

    def plot_confusion_matrix(self):
        cm = self.build_confusion_matrix()
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar(cax)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center', color='red')

        plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
        plt.close()

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Training Loss')
        plt.plot(self.validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
        plt.close()

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_metrics, label='Training Accuracy')
        plt.plot(self.validation_metrics, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
        plt.close()

    def export(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        self.plot_loss()
        self.plot_metrics()
        self.plot_confusion_matrix()