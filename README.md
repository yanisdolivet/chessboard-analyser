# My Torch - Chess Position Analyzer

![Status](https://img.shields.io/badge/Status-Development-yellow)
![Python](https://img.shields.io/pypi/pyversions/3)
![Build](https://img.shields.io/badge/Build-Makefile-green)

---

## About the project

A neural network-based chess position analyzer built from scratch in Python using only NumPy. This project implements a custom deep learning framework ("My Torch") to classify chess positions into three game states: Nothing (normal position), Check, or Checkmate.

### Key Features

- **Custom Neural Network Framework**: Fully implemented forward/backward propagation, gradient descent, and optimization techniques without using PyTorch or TensorFlow
- **FEN Position Parsing**: Converts chess positions from Forsyth-Edwards Notation (FEN) to one-hot encoded vectors
- **Advanced Training Features**:
  - Mini-batch gradient descent with momentum
  - L2 regularization for weight decay
  - Dropout for preventing overfitting
  - Learning rate decay
  - Early stopping with validation monitoring
- **Three-Class Classification**: Predicts game state (Nothing, Check, Checkmate) for any chess position
- **Comprehensive Analytics**: Built-in metrics tracking, confusion matrices, precision-recall curves, and loss visualization
- **Binary Network Format**: Custom binary protocol for saving/loading trained models with full configuration metadata

---

## Installation & Compilation

> Before diving in the setup of the project make sure you have at least Python3.11 installed with pip 22.3

1. Clone the repository
```bash
git clone git@github.com:yanisdolivet/chessboard-analyser.git
```

2. Create a python env (Optional)
```bash
python -m venv .venv
```

3. Install all dependencies
```bash
pip install -r ./requirements.txt
```

4. Compilation
```bash
make re
```

## User Guide

Once the project has correctly been setup we can move on the exucution of the binary

The project is split into 3 parts: The generator, the training and the predict.
The generator's purpose is to create blank neural network that we can use to train.

If you want to directly go to the train step you can. The project gives you 3 blank neural network (basic_network_0/1/2.nn).

> :warning: **If you changed the configuration of the neural network you may need to generate new blank neural network**


1. Create Blank Neural Network
```bash
$> ./my_torch_generator --help
USAGE
     ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
DESCRIPTION
     config_file_i   Configuration file containing description of a neural network we want to generate.
     nb_i            Number of neural networks to generate based on the configuration file.

$> ./my_torch_generator ./config/basic_network.json 1
```

After the generation of a blank neural network what we want is to train him to understand chess position. Right now he will just make random guess.

In order to trai him you have to give him a dataset that he can learn from. We aren't providing any due to the size of each of them that would be too large for the repository. For that you can go to this link: **[basic-dataset](https://huggingface.co/datasets/bonna46/Chess-FEN-and-NL-Format-30K-Dataset)** and then create a small python program to format  it like this:
r1bq1b1r/3n1kp1/1pnppp1p/p1pPN3/P5P1/N1PK4/1P1QPP1P/R1B2B1R b - - 2 14 Check

Once you've got your data set you can move on part 2

2. Train a neural network
```bash
$> ./my_torch_analyzer
usage: ./my_torch_analyzer.py [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

positional arguments:
  LOADFILE
  CHESSFILE

options:
  -h, --help   show this help message and exit
  --train
  --predict
  --save SAVE

$> ./my_torch_analyzer --train --save my_torch_network_test.nn basic_network_0.nn ./your/path/to/your/dataset.txt
```

> :warning: **The training part can take quite a while depending on your dataset size. For a referencial of time, with 20k positions in the dataset it takes, for us, ~15 mintues.**

Once we've train our network we can predict a given postion, using a file with the same syntax as the training dataset (FEN + result)

3. Predict
```bash
./my_torch_analyzer --predict my_torch_network_test.nn ./path/to/your/predict/file.txt
```

---

## TECHNICAL PART

### Architecture

- **Input Layer**: 2305 neurons (64 squares × 12 piece types x 3 states + 1 neuron fro White/Black to move)
- **Hidden Layers**: Configurable architecture with ReLU activation
- **Output Layer**: 3 neurons with softmax activation
- **Optimization**: L2 regularization (=0.001) with He/Xavier weight initialization

### Training Dataset

Trained on large-scale chess position datasets with balanced class distribution:
- 40% Nothing positions
- 40% Check positions
- 20% Checkmate positions

### Project Structure

```
├── src/
│   ├── my_torch/          # Neural network implementation
│   │   ├── Network.py     # Multi-layer network with training loop
│   │   └── Layer.py       # Single layer with forward/backward pass
│   ├── analyzer/          # Chess-specific components
│   │   ├── FENParser.py   # FEN notation parser
│   │   └── ModelLoader.py # Binary model loader
│   └── data_analysis/     # Metrics and visualization
├── tools/
│   ├── generator/         # Network generator from config
│   ├── balance_dataset.py # Dataset balancing utility
│   └── verify_encoding.py # FEN encoding verification
├── config/                # Network configuration files
└── data/                  # Training/validation datasets
```

### Technologies

- **Language**: Python 3
- **Core Library**: NumPy (matrix operations only)
- **Visualization**: Matplotlib (for metrics)
- **Documentation**: Sphinx with Google-style docstrings

---

## Performance & Benchmarking

The neural network's performance has been extensively evaluated across multiple metrics and training configurations. Our analysis includes training convergence patterns, loss curves, accuracy evolution, and per-class performance metrics through precision-recall curves and confusion matrices. The model demonstrates strong classification capabilities, particularly in distinguishing critical game states like checkmate from normal positions. Comprehensive benchmarking results, including detailed performance comparisons across different hyperparameter configurations and training dataset sizes, can be found in our dedicated **[Performance Analysis Documentation](./docs/performance-analysis.md)**.

---

## More Informations

For more informations about architecture of the project, the math foundatation and the optimization strategy please refer to the **[Technical Report](./docs/technical-report.md)**

---

*Document Version: 1.0*
*Published: December 21, 2025*
*Author: Yanis Mignot, Yanis Dolivet*

