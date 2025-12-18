import matplotlib.pyplot as plt
import numpy as np
import os
import random

# CONFIGURATION
EPOCHS = 50
OUTPUT_DIR = "out/curves"
OUTPUT_FILE = "training_metrics.png"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_mock_data(epochs):
    """
    Generates realistic-looking mock data for ML training.
    """
    x = np.arange(1, epochs + 1)

    # Simulate Loss: Exponential decay + random noise
    # Formula: y = a * exp(-b * x) + noise
    base_loss = 2.5 * np.exp(-0.1 * x)
    noise_loss = np.random.normal(0, 0.05, epochs)
    loss = base_loss + noise_loss
    loss = np.maximum(loss, 0)  # Ensure loss doesn't go below 0

    # Simulate Accuracy: Inverse exponential growth + random noise
    # Formula: y = 1 - (a * exp(-b * x)) + noise
    base_acc = 1 - (0.9 * np.exp(-0.08 * x))
    noise_acc = np.random.normal(0, 0.02, epochs)
    accuracy = base_acc + noise_acc
    accuracy = np.clip(accuracy, 0, 1)  # Ensure accuracy stays between 0 and 1

    return x, loss, accuracy


def plot_metrics(epochs, loss, accuracy):
    """
    Plots the training curves and saves them to a file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Loss
    ax1.plot(epochs, loss, "r-", label="Training Loss")
    ax1.set_title("Mock Training Loss (Categorical Cross-Entropy)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(epochs, accuracy, "b-", label="Training Accuracy")
    ax2.set_title("Mock Training Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Save figure
    ensure_dir(OUTPUT_DIR)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(save_path)
    print(f"Graph saved successfully to: {save_path}")

    # Close plot to free memory
    plt.close()


if __name__ == "__main__":
    print(f"Starting Mock Training Simulation for {EPOCHS} epochs...")

    # 1. Generate Fake Data
    epochs_range, loss_data, acc_data = generate_mock_data(EPOCHS)

    # 2. Print last values to simulate training output
    print(
        f"Final Epoch {EPOCHS}: Loss={loss_data[-1]:.4f}, Accuracy={acc_data[-1]*100:.2f}%"
    )

    # 3. Plot and Export
    plot_metrics(epochs_range, loss_data, acc_data)
