import numpy as np
import sys

ERROR_CODE = 84

class FENParser:
    """
    Gère le parsing des fichiers FEN pour générer des tableaux NumPy
    d'entrée (X) et de sortie (Y) pour l'entraînement du réseau neuronal.
    """

    PIECE_INDEX = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    INPUT_SIZE = 768

    OUTPUT_SIZE = 3

    def __init__(self):
        pass

    def _encode_board_position(self, board_position: str) -> np.ndarray:
        """
        Convertit la partie de position FEN en un vecteur One-Hot de 768 dimensions.
        """
        board_vector = []

        for char in board_position:

            if char == '/':
                continue

            elif '1' <= char <= '8':
                nb_empty_squares = int(char)
                board_vector.extend([0.0] * (nb_empty_squares * 12))

            elif char in self.PIECE_INDEX:
                index = self.PIECE_INDEX[char]
                one_hot_segment = [0.0] * 12
                one_hot_segment[index] = 1.0
                board_vector.extend(one_hot_segment)

        if len(board_vector) != self.INPUT_SIZE:
            print(f"Warning: FEN produced {len(board_vector)} features, expected {self.INPUT_SIZE}.", file=sys.stderr)
            return np.array([])

        return np.array(board_vector, dtype=np.float32)

    def _map_result_to_one_hot(self, result_word: str) -> np.ndarray:
        """
        Convertit le résultat ('Nothing', 'Check', 'Checkmate') en vecteur One-Hot 1x3.
        """
        if result_word == "Nothing":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif result_word == "Check":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif result_word == "Checkmate":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            print(f"Warning: Unknown result word '{result_word}'. Mapping to Nothing.", file=sys.stderr)
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def parse_file(self, chessfile: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Lit le fichier FEN ligne par ligne et retourne X (entrées) et Y (cibles) en tableaux NumPy.
        """
        input_data = []
        output_targets = []

        try:
            with open(chessfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()

                    board_position_part = parts[0]

                    if len(parts) >= 2 and parts[-1] in ["Nothing", "Check", "Checkmate"]:
                        result_word = parts[-1]
                    else:
                        result_word = "Nothing"

                    board_fen = board_position_part.split(' ')[0]

                    X_vector = self._encode_board_position(board_fen)

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