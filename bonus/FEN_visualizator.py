#!/usr/bin/env python3

##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## FEN_visualizator
##

import sys
from tkinter import Tk, Canvas, PhotoImage
from PIL import Image, ImageTk
import os

black_pieces = {
    'r', 'n', 'b', 'q', 'k', 'p'
}
white_pieces = {
    'R', 'N', 'B', 'Q', 'K', 'P'
}

def fen_to_board(fen):
    """Transform fen string notation to a chess board

    Args:
        fen (str): FEN string notation of the chess board.

    Returns:
        list: 2D list representing the chess board.

    Example:
        >>> board = fen_to_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") # Returns starting position board
    """
    rows = fen.split(' ')[0].split('/')
    board = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    board_row.append('.')
            else:
                board_row.append(char)
        board.append(board_row)
    return board

def get_piece_image(piece):
    """Get the cropped image of a chess piece from combined sprite sheet.

    The sprite sheet contains pieces in a 2x6 grid:
    - Top row (black): Rook, Knight, Bishop, Queen, King, Pawn
    - Bottom row (white): Rook, Knight, Bishop, Queen, King, Pawn

    Args:
        piece (str): Single character representing the piece.
            Uppercase for white pieces: K, Q, R, B, N, P
            Lowercase for black pieces: k, q, r, b, n, p

    Returns:
        PIL.Image: Cropped image of the piece, or None if piece is empty ('.')

    Example:
        >>> img = get_piece_image('K')  # Returns white king image
        >>> img = get_piece_image('p')  # Returns black pawn image
    """
    if piece == '.':
        return None

    is_black = piece.islower()
    piece_map = {
        'R': 0, 'r': 0,  # Rook
        'N': 1, 'n': 1,  # Knight
        'B': 2, 'b': 2,  # Bishop
        'Q': 3, 'q': 3,  # Queen
        'K': 4, 'k': 4,  # King
        'P': 5, 'p': 5   # Pawn
    }

    if piece not in piece_map:
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sprite_path = os.path.join(script_dir, "../resources/chess.pieces.png")

    sprite_sheet = Image.open(sprite_path)

    sheet_width, sheet_height = sprite_sheet.size
    piece_width = sheet_width // 6
    piece_height = sheet_height // 2

    piece_col = piece_map[piece]

    left = piece_col * piece_width
    top = 0 if is_black else piece_height
    right = left + piece_width
    bottom = top + piece_height

    piece_image = sprite_sheet.crop((left, top, right, bottom))

    return piece_image


def visualize_board(board):
    """Visualize a chess board using Tkinter.

    Args:
        board (list): 2D list representing the chess board.
            Each element is a piece character or '.' for empty.

    Example:
        >>> board = fen_to_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        >>> visualize_board(board)
    """
    square_size = 125
    root = Tk()
    root.title("Chess Board Visualizer")
    canvas = Canvas(root, width=square_size*8, height=square_size*8)
    canvas.pack()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    chessboard_path = os.path.join(script_dir, "../resources/chessboard.png")
    chessboard_img = Image.open(chessboard_path)
    chessboard_img = chessboard_img.resize((square_size*8, square_size*8), Image.LANCZOS)
    chessboard_photo = ImageTk.PhotoImage(chessboard_img)
    canvas.create_image(0, 0, anchor='nw', image=chessboard_photo)

    canvas.image = chessboard_photo
    canvas.piece_images = []

    for row_idx, row in enumerate(board):
        for col_idx, piece in enumerate(row):
            if piece != '.':
                piece_img = get_piece_image(piece)
                if piece_img:

                    piece_img = piece_img.resize((square_size, square_size), Image.LANCZOS)
                    piece_photo = ImageTk.PhotoImage(piece_img)

                    x = col_idx * square_size + square_size // 2
                    y = row_idx * square_size + square_size // 2

                    canvas.create_image(x, y, image=piece_photo)

                    canvas.piece_images.append(piece_photo)

    root.mainloop()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 FEN_visualizator.py FEN_file_string")
        return
    fen = sys.argv[1]
    with open(fen, 'r') as file:
        fen = file.read().strip()
    for line in fen.split('\n'):
        board = fen_to_board(line)
        for row in board:
            print(' '.join(row))
        print()
        visualize_board(board)
if (__name__ )== "__main__":
    main()