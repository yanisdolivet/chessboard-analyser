#!/bin/python

import json

def process_chess_data(input_file, output_file):
    """Process chess data from a JSONL file and write FEN with state to output.

    Args:
        input_file (str): _path to input JSONL file
        output_file (str): _path to output text file
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Skip empty lines
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Extract necessary fields
                fen = data.get("fen", "")
                move_san = data.get("move_san", "")
                # Determine the suffix based on the move annotation
                suffix = "Nothing"
                if "#" in move_san:
                    suffix = "Checkmate"
                elif "+" in move_san:
                    suffix = "Check"
                # Format the output line
                output_line = f"{fen} {suffix}\n"
                f_out.write(output_line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

process_chess_data('./data/train.jsonl', './data/output.txt')