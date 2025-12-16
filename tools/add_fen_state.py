"""Add a 'State' column to a CSV containing FEN positions.

This script reads a CSV with a header containing a `FEN` column and writes
a new CSV with an added `State` column whose values are one of:
 - Nothing
 - Check
 - Checkmate

It creates a backup of the original file with a .bak extension.

Usage:
    python tools/add_fen_state.py path/to/Chess_FEN%2BNL_format.csv

If python-chess is not installed this script will print an instruction.
"""
import sys
import os
import csv

try:
    import chess
except Exception:
    print("python-chess not installed. Install with: pip install python-chess")
    sys.exit(2)


def fen_state(fen):
    """Return 'Nothing', 'Check' or 'Checkmate' for the side to move in FEN."""
    board = chess.Board(fen)
    if board.is_checkmate():
        return "Checkmate"
    if board.is_check():
        return "Check"
    return "Nothing"


def process_file(path):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return 1

    # Read original CSV but do not modify it. We'll only produce a .txt output.
    with open(path, newline='') as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])

        # Prefer to replace a comment/description column with the State column.
        # Common comment-like headers in this dataset: 'NL Format of FEN', 'FEN + Next move', 'FEN state'
        comment_candidates = ['NL Format of FEN', 'FEN + Next move', 'FEN state', 'Comment', 'Description']
        replace_col = None
        for c in comment_candidates:
            if c in fieldnames:
                replace_col = c
                break

        # prepare TXT output path (same base, .txt)
        base, _ = os.path.splitext(path)
        txt_path = base + '.txt'
        with open(txt_path, 'w', newline='') as txt_fh:
            for row in reader:
                fen = row.get('FEN')
                # compute state
                if not fen:
                    state_value = ''
                else:
                    try:
                        state_value = fen_state(fen)
                    except Exception as e:
                        state_value = f'ERROR: {e}'

                # write TXT line: FEN NextMove State (space-separated). prefer 'Next move' header
                next_move = row.get('Next move') or row.get('FEN + Next move') or ''
                # normalize spacing and remove commas
                fen_txt = fen.replace(',', '') if fen else ''
                next_txt = next_move.replace(',', '').strip()
                state_txt = state_value.replace(',', '')
                txt_line = f"{fen_txt} {next_txt} {state_txt}\n"
                txt_fh.write(txt_line)

    print(f"Processed. TXT written to: {txt_path}")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/add_fen_state.py path/to/CSV')
        sys.exit(2)
    sys.exit(process_file(sys.argv[1]))
