##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## builder
###

#!/usr/bin/env python3

import sys, json, struct

MAGIC_NUMBER = 0x48435254  # 'TRCH' in hex

def save_network(filepath, layer_sizes, weights, biases):
    with open(filepath, 'wb') as f:
        # Header: Magic Num + Num of layers
        f.write(struct.pack('II', MAGIC_NUMBER, len(layer_sizes)))

        # Write layer sizes
        f.write(struct.pack(f'{len(layer_sizes)}I', *layer_sizes))

        # Write weights and biases
        f.write(struct.pack(f'{len(weights)}f', *weights))
        f.write(struct.pack(f'{len(biases)}f', *biases))

    print(f"Saved binary network to {filepath}")

def create_weights(init_method, nblayers):
    weights = []
    match init_method:
        case "he_normal":
            # He normal initialization using np
            pass
        case "random":
            # Random initialization using np
            pass
        case _:
            print(f"Unknown initialization method: {init_method}")
    return weights

def create_biases(nblayers):
    biases = [0.0] * nblayers
    return biases

def parse_config(filepath):
    layer_size = []
    init_method = "random"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            for layer in config.get("layers", []):
                size = layer.get("size", 0)
                layer_size.append(size)
            init_method = config.get("hyperparameters", {}).get("initialization", "random")
        return layer_size, init_method
    except Exception as e:
        print(f"Error reading config file: {e}")

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "default_path"
    print(f"Building project with configuration from: {filepath}")

    layer_size, init_method = parse_config(filepath)
    print("Configuration File Parsed Successfully")

    weights = create_weights(init_method, len(layer_size))
    biases = create_biases(len(layer_size))
    save_network("untrained_network.nn", layer_size, weights, biases)
    print(f"Untrained neural network successfully created with {len(layer_size)} layers.")