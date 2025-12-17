#!/usr/bin/env python3

import argparse
import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_py.analyzer.FENParser import FENParser
from src_py.analyzer.ModelLoader import ModelLoader
from src_py.my_torch.Network import Network

ERROR_CODE = 84


def parse_arguments():
    """
    Configure et analyse les arguments de la ligne de commande.
    Retourne l'objet args ou quitte avec le code d'erreur 84 en cas de conflit.
    """
    parser = argparse.ArgumentParser(
        description="""
        Analysez l'état d'un échiquier en utilisant un réseau neuronal entraîné.
        Le programme doit utiliser une solution basée sur l'apprentissage automatique
        avec apprentissage supervisé.
        """,
        usage="./my_torch_analyzer.py [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Lance le réseau neuronal en mode entraînement. Le CHESSFILE doit contenir les inputs FEN et l'output attendu.",
    )
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Lance le réseau neuronal en mode prédiction. Le CHESSFILE doit contenir les inputs FEN.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Sauvegarde le réseau entraîné dans SAVEFILE. Fonctionne uniquement en mode --train.",
    )

    parser.add_argument(
        "LOADFILE",
        type=str,
        help="Fichier contenant un réseau neuronal artificiel pré-entraîné ou à entraîner.",
    )

    parser.add_argument(
        "CHESSFILE", type=str, help="Fichier contenant les échiquiers en notation FEN."
    )

    args = parser.parse_args()

    if args.train:
        loadfile = args.LOADFILE
        chessfile = args.CHESSFILE
        savefile = args.save if args.save else loadfile

        if args.save:
            if len(sys.argv) != 6:
                print(
                    "Error: Invalid number of arguments for --train --save mode. Expected: 6.",
                    file=sys.stderr,
                )
                sys.exit(ERROR_CODE)
        elif len(sys.argv) != 4:
            print(
                "Error: Invalid number of arguments for --train mode. Expected: 4.",
                file=sys.stderr,
            )
            sys.exit(ERROR_CODE)

    elif args.predict:
        loadfile = args.LOADFILE
        chessfile = args.CHESSFILE
        savefile = None
        if args.save is not None:
            print("Error: --save is only allowed in --train mode.", file=sys.stderr)
            sys.exit(ERROR_CODE)
        if len(sys.argv) != 4:
            print(
                "Error: Invalid number of arguments for --predict mode. Expected: 4.",
                file=sys.stderr,
            )
            sys.exit(ERROR_CODE)

    if not os.path.exists(loadfile):
        print(f"Error: LOADFILE '{loadfile}' not found.", file=sys.stderr)
        sys.exit(ERROR_CODE)
    if not os.path.exists(chessfile):
        print(f"Error: CHESSFILE '{chessfile}' not found.", file=sys.stderr)
        sys.exit(ERROR_CODE)

    return args.train, loadfile, chessfile, savefile


def main():
    try:
        is_train, loadfile, chessfile, savefile = parse_arguments()

        parser = FENParser()
        X_data, Y_targets = parser.parse_file(chessfile)

        if len(X_data) == 0:
            print("Error: No valid FEN data found in the file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        print(f"Data loaded: {len(X_data)} examples.")

        loader = ModelLoader()
        layer_sizes, weights, biases = loader.load_network(loadfile)

        network = Network(layer_sizes, X_data, Y_targets)
        network.createLayer(weights, biases)

        if is_train:
            network.train(0.01, savefile)
        else:
            network.predict()

    except SystemExit:
        if sys.exc_info()[1].code != 0:
            sys.exit(ERROR_CODE)


if __name__ == "__main__":
    main()
