##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## data_analysis
##

import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

OUTPUT_DIR = "docs/benchmarks/"
VALIDATION_LOSS_FILE = os.path.join(OUTPUT_DIR, "validation_loss.txt")
TRAINING_LOSS_FILE = os.path.join(OUTPUT_DIR, "training_loss.txt")
VALIDATION_METRICS_FILE = os.path.join(OUTPUT_DIR, "validation_metrics.txt")
TRAINING_METRICS_FILE = os.path.join(OUTPUT_DIR, "training_metrics.txt")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.txt")
EXPECTED_FILE = os.path.join(OUTPUT_DIR, "expected.txt")

FILE = [
    VALIDATION_LOSS_FILE,
    TRAINING_LOSS_FILE,
    VALIDATION_METRICS_FILE,
    TRAINING_METRICS_FILE,
    PREDICTIONS_FILE,
    EXPECTED_FILE,
]


class DataAnalysis:
    def __init__(self, modelspec=None, epochs=100):
        self.training_loss = []
        self.validation_loss = []

        self.training_metrics = []
        self.validation_metrics = []
        self.epochs = epochs

        self.prediction = []
        self.expected = []

        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        try:
            for file in FILE:
                with open(file, "a") as f:
                    f.write(f"\n\n#############################\n")
                    f.write(f"# New run start at {self.date}\n")
                    f.write(f"# Model Specifications:\n")
                    for layer in range(modelspec.num_layers):
                        f.write(
                            f"# Layer {layer}: Type={modelspec.type[layer]}, Size={modelspec.layer_sizes[layer]}, Activation={modelspec.activation[layer]}\n"
                        )
                    f.write(f"# Hyperparameters:\n")
                    f.write(
                        f"# Learning Rate={modelspec.learning_rate}, Initialization={modelspec.initialization}\n"
                    )
                    f.write(f"# Training Parameters:\n")
                    f.write(
                        f"# Batch Size={modelspec.batch_size}, Lreg={modelspec.lreg}, Dropout Rate={modelspec.dropout_rate}, Epochs={modelspec.epochs}, Loss Function={modelspec.loss_function}\n"
                    )
                    f.write(f"#############################\n")
        except Exception as e:
            print(f"Error initializing data analysis files: {e}")

    def save_loss(self, training_loss, validation_loss):
        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)
        with open(TRAINING_LOSS_FILE, "a") as f:
            f.write(f"{training_loss}\n")
        with open(VALIDATION_LOSS_FILE, "a") as f:
            f.write(f"{validation_loss}\n")

    def save_metrics(self, training_metric, validation_metric):
        self.training_metrics.append(training_metric)
        self.validation_metrics.append(validation_metric)
        with open(TRAINING_METRICS_FILE, "a") as f:
            f.write(f"{training_metric}\n")
        with open(VALIDATION_METRICS_FILE, "a") as f:
            f.write(f"{validation_metric}\n")

    def save_predictions(self, preds, truths):
        self.prediction.extend(preds)
        self.expected.extend(truths)
        with open(PREDICTIONS_FILE, "a") as f:
            for pred in preds:
                f.write(f"{pred}\n")
        with open(EXPECTED_FILE, "a") as f:
            for truth in truths:
                f.write(f"{truth}\n")

    def build_confusion_matrix(self):
        num_classes = 3
        confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

        for true, pred in zip(self.expected, self.prediction):
            confusion_matrix[true][pred] += 1

        return confusion_matrix

    def calculate_precision_recall(self):
        cm = self.build_confusion_matrix()
        num_classes = len(cm)
        precision = []
        recall = []

        for i in range(num_classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(num_classes) if j != i)
            fn = sum(cm[i][j] for j in range(num_classes) if j != i)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision.append(prec)
            recall.append(rec)

        return precision, recall

    def plot_precision_recall(self):
        savedir = os.path.join(OUTPUT_DIR, "precision_recall")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        precision, recall = self.calculate_precision_recall()
        classes = [f"Class {i}" for i in range(len(precision))]

        x = np.arange(len(classes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, precision, width, label="Precision")
        rects2 = ax.bar(x + width / 2, recall, width, label="Recall")

        ax.set_ylabel("Scores")
        ax.set_title("Precision and Recall by Class")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        plt.savefig(os.path.join(savedir, f"precision_recall_{self.date}.png"))
        plt.close()

    def plot_confusion_matrix(self):
        savedir = os.path.join(OUTPUT_DIR, "confusion_matrix")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        cm = self.build_confusion_matrix()
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar(cax)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha="center", va="center", color="red")

        plt.savefig(os.path.join(savedir, f"confusion_matrix_{self.date}.png"))
        plt.close()

    def plot_loss(self):
        savedir = os.path.join(OUTPUT_DIR, "loss")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label="Training Loss")
        plt.plot(self.validation_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"loss_curve_{self.date}.png"))
        plt.close()

    def plot_metrics(self):
        savedir = os.path.join(OUTPUT_DIR, "metrics")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plt.figure(figsize=(10, 5))
        plt.plot(self.training_metrics, label="Training Accuracy")
        plt.plot(self.validation_metrics, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"accuracy_curve_{self.date}.png"))
        plt.close()

    def export(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        self.plot_loss()
        self.plot_metrics()
        self.plot_confusion_matrix()
        self.plot_precision_recall()
