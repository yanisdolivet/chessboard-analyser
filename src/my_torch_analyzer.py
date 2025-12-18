#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np  # INDISPENSABLE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer.FENParser import FENParser
from src.analyzer.ModelLoader import ModelLoader
from src.my_torch.Network import Network

ERROR_CODE = 84

def parse_arguments():
    parser = argparse.ArgumentParser(usage="./my_torch_analyzer.py [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--predict", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("LOADFILE", type=str)
    parser.add_argument("CHESSFILE", type=str)
    args = parser.parse_args()

    if args.train:
        savefile = args.save if args.save else args.LOADFILE
    else:
        savefile = None

    return args.train, args.LOADFILE, args.CHESSFILE, savefile

def main():
    try:
        is_train, loadfile, chessfile, savefile = parse_arguments()

        parser = FENParser()
        X_data, Y_targets = parser.parse_file(chessfile)

        print(f"First 5 samples of X data:\n{X_data[:5]}")
        print(f"First 5 samples of Y targets:\n{Y_targets[:5]}")

        if len(X_data) == 0:
            print("Error: No valid FEN data.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        print(f"Data loaded: {len(X_data)} samples.")

        loader = ModelLoader()
        layer_sizes, weights, biases = loader.load_network(loadfile)

        if is_train:
            indices = np.arange(len(X_data))
            np.random.shuffle(indices)
            X_data, Y_targets = X_data[indices], Y_targets[indices]

            split = int(0.8 * len(X_data))
            X_train, X_val = X_data[:split], X_data[split:]
            Y_train, Y_val = Y_targets[:split], Y_targets[split:]

            print(f"Training set: {len(X_train)} | Validation set: {len(X_val)}")

            network = Network(layer_sizes, X_train, Y_train)
            network.createLayer(weights, biases)

            network.train(0.05, savefile, X_val=X_val, Y_val=Y_val)

        else:
            print("Running prediction...")
            network = Network(layer_sizes, X_data, Y_targets)
            network.createLayer(weights, biases)
            network.predict()

    except SystemExit:
        sys.exit(ERROR_CODE)

if __name__ == "__main__":
    main()