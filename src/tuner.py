#!/usr/bin/env python3

import optuna
import sys
import os
import numpy as np
from src.analyzer.FENParser import FENParser
from src.my_torch.Network import Network
from src.my_torch.Layer import Layer

ERROR_CODE = 84

# On charge les données UNE SEULE FOIS au début pour gagner du temps
if len(sys.argv) < 2:
    print("Usage: python3 tuner.py DATA_FILE")
    sys.exit(ERROR_CODE)

DATA_FILE = sys.argv[1]
parser = FENParser()
print(f"Chargement des données depuis {DATA_FILE}...")
X_ALL, Y_ALL = parser.parse_file(DATA_FILE)

if len(X_ALL) == 0:
    print("Erreur : Aucune donnée chargée.")
    sys.exit(ERROR_CODE)

def objective(trial):
    # --- 1. Paramètres à tester ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout", 0.05, 0.3)
    
    # On teste différentes largeurs de couches
    n1 = trial.suggest_int("n_hidden1", 256, 1024, step=128)
    n2 = trial.suggest_int("n_hidden2", 64, 512, step=64)
    topology = [769, n1, n2, 3]

    # --- 2. Initialisation ---
    network = Network(topology, X_ALL, Y_ALL)
    for i in range(1, len(topology)):
        l1, l2 = topology[i-1], topology[i]
        layer = Layer(l1, l2, dropout_rate=dropout_rate)
        # He initialization
        layer.weights = np.random.randn(l1, l2) * np.sqrt(2.0 / l1)
        layer.biases = np.zeros((1, l2))
        network.layers.append(layer)

    # --- 3. Entraînement rapide (5 époques suffisent pour comparer) ---
    num_epochs = 5
    num_samples = len(X_ALL)
    
    for epoch in range(num_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_s, Y_s = X_ALL[indices], Y_ALL[indices]

        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            x_b, y_b = X_s[i:i+batch_size], Y_s[i:i+batch_size]
            pred = network.forward(x_b, training=True)
            
            # Cross-Entropy (moyenne)
            loss = -np.mean(np.sum(y_b * np.log(pred + 1e-15), axis=1))
            epoch_loss += loss * len(x_b)
            
            grad = (pred - y_b) / len(x_b)
            network.backward(grad, lr)
        
        current_avg_loss = epoch_loss / num_samples
        
        # On rapporte la perte à Optuna pour le pruning
        trial.report(current_avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return current_avg_loss

def main():
    # On peut augmenter n_trials si on a le temps
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n🚀 OPTIMISATION TERMINÉE")
    print(f"Meilleure Loss trouvée : {study.best_value:.6f}")
    print("Meilleure configuration :")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")

if __name__ == "__main__":
    main()