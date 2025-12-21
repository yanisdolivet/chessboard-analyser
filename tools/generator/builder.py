#!/usr/bin/env python3

##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## builder
###

import sys, json, struct
import numpy as np

MAGIC_NUMBER = 0x48435254  # 'TRCH' in hex
VERSION = 2  # Binary protocol version


def encode_string(s):
    """Encode string as length-prefixed bytes."""
    encoded = s.encode("utf-8")
    return struct.pack("I", len(encoded)) + encoded


def save_network(filepath, layer_sizes, weights, biases, config):
    """Save neural network to binary file with full configuration.

    Args:
        filepath (str): Path to save the network file.
        layer_sizes (list): List of integers representing layer sizes.
        weights (list): List of numpy arrays containing weight matrices.
        biases (list): List of numpy arrays containing bias vectors.
        config (dict): Configuration dictionary with all network parameters.
    """
    with open(filepath, "wb") as f:
        # Header: Magic Num + Version + Num of layers
        f.write(struct.pack("III", MAGIC_NUMBER, VERSION, len(layer_sizes)))

        # Write layer sizes
        f.write(struct.pack(f"{len(layer_sizes)}I", *layer_sizes))

        # Write layer types and activations
        for layer in config.get("layers", []):
            layer_type = layer.get("type", "HIDDEN")
            activation = layer.get("activation", "none")
            f.write(encode_string(layer_type))
            f.write(encode_string(activation))

        # Write hyperparameters
        hyperparams = config.get("hyperparameters", {})
        learning_rate = hyperparams.get("learning_rate", 0.01)
        initialization = hyperparams.get("initialization", "random")
        f.write(struct.pack("f", learning_rate))
        f.write(encode_string(initialization))

        # Write training parameters
        training = config.get("training", {})
        batch_size = training.get("batch_size", 32)
        epochs = training.get("epochs", 100)
        lreg = training.get("lreg", 0.0)
        loss_function = training.get("loss_function", "mse")
        dropout_rate = training.get("dropout_rate", 0.0)

        f.write(struct.pack("I", batch_size))
        f.write(struct.pack("I", epochs))
        f.write(struct.pack("f", lreg))
        f.write(struct.pack("f", dropout_rate))
        f.write(encode_string(loss_function))

        # Write weights and biases as floats
        for w in weights:
            w_flat = w.flatten()
            f.write(struct.pack(f"{len(w_flat)}f", *w_flat))

        for b in biases:
            b_flat = b.flatten()
            f.write(struct.pack(f"{len(b_flat)}f", *b_flat))

    print(f"Saved binary network to {filepath}")


def he_normal(layer_size, seed=0):
    weights = []
    np.random.seed(seed)
    L = len(layer_size)
    for l in range(1, L):
        stddev = np.sqrt(2.0 / layer_size[l - 1])
        w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
        weights.append(w)
    return weights


def create_weights(init_method, layer_size):
    match init_method:
        case "he_normal":
            weights = he_normal(layer_size)
        case "xavier":
            weights = []
            L = len(layer_size)
            for l in range(1, L):
                stddev = np.sqrt(1.0 / layer_size[l - 1])
                w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
                weights.append(w)
        case "he_mixed_xavier":
            weights = []
            L = len(layer_size)
            for l in range(1, L):
                if l == L - 1:
                    stddev = np.sqrt(1.0 / layer_size[l - 1])
                else:
                    stddev = np.sqrt(2.0 / layer_size[l - 1])
                w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
                weights.append(w)
        case "random":
            weights = [
                np.random.rand(layer_size[l], layer_size[l - 1]) * 0.01
                for l in range(1, len(layer_size))
            ]
        case _:
            print(f"Unknown initialization method: {init_method}")
    return weights


def create_biases(layer_size):
    biases = [np.zeros((layer_size[l], 1)) for l in range(1, len(layer_size))]
    return biases


def parse_config(filepath):
    layer_size = []
    init_method = "random"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
            for layer in config.get("layers", []):
                size = layer.get("size", 0)
                print(type(size))
                layer_size.append(size)
            init_method = config.get("hyperparameters", {}).get(
                "initialization", "random"
            )
        return layer_size, init_method, config
    except Exception as e:
        print(f"Error reading config file: {e}")


def print_usage():
    print("USAGE")
    print("     ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]")
    print("DESCRIPTION")
    print(
        "     config_file_i   Configuration file containing description of a neural network we want to generate."
    )
    print(
        "     nb_i            Number of neural networks to generate based on the configuration file."
    )


def error_args():
    if len(sys.argv) <= 2:
        if "--help" in sys.argv or "-h" in sys.argv:
            print_usage()
        else:
            print("Error: Invalid number of arguments.")
            print_usage()
        sys.exit(0)
    elif len(sys.argv) < 2:
        print("Error: Invalid number of arguments.")
        print_usage()
        sys.exit(1)


def main():
    error_args()
    filepath, nbofnn = sys.argv[1], (
        int(sys.argv[2]) if sys.argv[2].isdigit() else print_usage() or sys.exit(1)
    )
    print(f"Building project with configuration from: {filepath}")

    layer_size, init_method, config = parse_config(filepath)
    print("Configuration File Parsed Successfully")

    for nn in range(nbofnn):
        nnfilepath = f"{filepath.split('/')[-1].split('.')[0]}_{nn}.nn"
        weights = create_weights(init_method, layer_size)
        biases = create_biases(layer_size)
        save_network(nnfilepath, layer_size, weights, biases, config)
        print(
            f"Untrained neural network successfully created with {len(layer_size)} layers."
        )


if __name__ == "__main__":
    main()
