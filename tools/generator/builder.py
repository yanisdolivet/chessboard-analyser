##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## builder
###

#!/usr/bin/env python3

import sys, json

def parse_config(filepath):
    layer_size = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
            for layer in config.get("layers", []):
                size = layer.get("size", 0)
                layer_size.append(size)
        return layer_size
    except Exception as e:
        print(f"Error reading config file: {e}")

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "default_path"
    print(f"Building project with configuration from: {filepath}")
    layer_size = parse_config(filepath)
    print(f"Layer size: {layer_size}")