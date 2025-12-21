import numpy as np
import sys
import chess

ERROR_CODE = 84


class FENParser:
    """
    Parser for FEN chess positions and their results.
    """

    PIECE_INDEX = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    INPUT_SIZE = 2305

    OUTPUT_SIZE = 3

    def __init__(self):
        pass

    def _encode_board_position(self, board_position: str) -> np.ndarray:
        """
        Convert the FEN board position part into a 768-dimensional One-Hot vector.
        Args:
            board_position (str): FEN string representing the board position.
        Returns:
            np.ndarray: 1D NumPy array of shape (2305,) representing the encoded
        """
        board = chess.Board(board_position)
        matrix = np.zeros((8, 8, 36), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        white_attack = np.zeros((8, 8), dtype=bool)
        black_attack = np.zeros((8, 8), dtype=bool)

        for square in chess.SQUARES:
            if board.is_attacked_by(chess.WHITE, square):
                white_attack[chess.square_rank(square)][
                    chess.square_file(square)
                ] = True
            if board.is_attacked_by(chess.BLACK, square):
                black_attack[chess.square_rank(square)][
                    chess.square_file(square)
                ] = True

        for square in chess.SQUARES:
            piece_on_board = board.piece_at(square)
            if piece_on_board:
                piece_idx = piece_map[piece_on_board.piece_type] + (
                    6 if piece_on_board.color == chess.BLACK else 0
                )
                base_channel = piece_idx * 3

                row = chess.square_rank(square)
                col = chess.square_file(square)
                matrix[row, col, base_channel] = 1.0

                if white_attack[row, col]:
                    matrix[row, col, base_channel + 1] = 1.0
                if black_attack[row, col]:
                    matrix[row, col, base_channel + 2] = 1.0

        flat_vector = matrix.flatten()
        turn_val = 1.0 if board.turn == chess.WHITE else 0.0
        return np.array(np.append(flat_vector, turn_val), dtype=np.float32)

    def _map_result_to_one_hot(self, result_word: str) -> np.ndarray:
        """
        Convert the result ('Nothing', 'Check', 'Checkmate') into a 1x3 One-Hot vector
        Args:
            result_word (str): Result string.
        Returns:
            np.ndarray: 1D NumPy array of shape (3,) representing the encoded result.
        """
        if result_word == "Nothing":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif result_word == "Check":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif result_word == "Checkmate":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            print(
                f"Warning: Unknown result word '{result_word}'. Mapping to Nothing.",
                file=sys.stderr,
            )
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def parse_file(self, chessfile: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Read the FEN file line by line and return X (inputs) and Y (targets) as NumPy arrays.
        Args:
            chessfile (str): Path to the FEN file.
        Returns:
            tuple: (X, Y) where X is the input data and Y is the target data.
        """
        input_data = []
        output_targets = []

        try:
            with open(chessfile, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) >= 2 and parts[-1] in [
                        "Nothing",
                        "Check",
                        "Checkmate",
                    ]:
                        result_word = parts[-1]
                    else:
                        result_word = "Nothing"

                    X_vector = self._encode_board_position(" ".join(parts[:-1]))

                    Y_vector = self._map_result_to_one_hot(result_word)

                    if X_vector.size == self.INPUT_SIZE:
                        input_data.append(X_vector)
                        output_targets.append(Y_vector)

        except FileNotFoundError:
            print(f"Error: Chessfile '{chessfile}' not found.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        X = np.array(input_data, dtype=np.float32)
        Y = np.array(output_targets, dtype=np.float32)

        return X, Y
