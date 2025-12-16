import numpy as np
import struct
import sys

MAGIC_NUMBER = 0x48435254
ERROR_CODE = 84

class ModelLoader:
    """
    Gère la lecture des fichiers binaires du réseau neuronal (.nn)
    et charge la topologie, les poids et les biais dans des tableaux NumPy.
    """

    def __init__(self):
        pass

    def _read_header(self, f) -> int:
        """Lit et valide le Magic Number et retourne le nombre de couches."""
        header_data = f.read(8)
        if len(header_data) < 8:
            print("Error: Invalid header size or empty file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        magic_number, layer_count = struct.unpack('II', header_data)

        if magic_number != MAGIC_NUMBER:
            print("Error: Invalid magic number in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        if layer_count < 2:
            print("Error: Network must have at least an input and an output layer.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        return layer_count

    def _read_layer_sizes(self, f, layer_count: int) -> list[int]:
        """Lit la taille de chaque couche (la topologie)."""
        topology_data = f.read(layer_count * 4)
        if len(topology_data) < layer_count * 4:
            print("Error: Missing layer size information.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        layer_sizes = list(struct.unpack(f'{layer_count}I', topology_data))
        return layer_sizes

    def _read_and_reshape_weights(self, f, layer_sizes: list[int]) -> list[np.ndarray]:
        """Lit le segment des poids, le déserialise et le remodèle en matrices."""
        weights = []
        layer_count = len(layer_sizes)

        total_weights = sum(layer_sizes[i-1] * layer_sizes[i] for i in range(1, layer_count))

        weight_data = f.read(total_weights * 4)
        if len(weight_data) < total_weights * 4:
            print("Error: Incomplete weight data in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        flat_weights = struct.unpack(f'{total_weights}f', weight_data)

        current_idx = 0
        for i in range(1, layer_count):
            input_size = layer_sizes[i-1]
            output_size = layer_sizes[i]
            matrix_size = input_size * output_size

            w_segment = flat_weights[current_idx : current_idx + matrix_size]

            w_matrix = np.array(w_segment, dtype=np.float32).reshape((input_size, output_size))

            weights.append(w_matrix)
            current_idx += matrix_size

        return weights

    def _read_and_reshape_biases(self, f, layer_sizes: list[int]) -> list[np.ndarray]:
        """Lit le segment des biais, le déserialise et le remodèle en vecteurs lignes."""
        biases = []
        layer_count = len(layer_sizes)

        total_biases = sum(layer_sizes[1:])

        bias_data = f.read(total_biases * 4)
        if len(bias_data) < total_biases * 4:
            print("Error: Incomplete bias data in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        flat_biases = struct.unpack(f'{total_biases}f', bias_data)

        current_idx = 0
        for i in range(1, layer_count):
            output_size = layer_sizes[i]

            b_segment = flat_biases[current_idx : current_idx + output_size]

            b_vector = np.array(b_segment, dtype=np.float32).reshape((1, output_size))

            biases.append(b_vector)
            current_idx += output_size

        return biases

    def load_network(self, filepath: str) -> tuple[list, list, list]:
        """Fonction principale orchestrant la lecture du fichier .nn."""

        try:
            with open(filepath, 'rb') as f:

                layer_count = self._read_header(f)
                layer_sizes = self._read_layer_sizes(f, layer_count)
                weights = self._read_and_reshape_weights(f, layer_sizes)
                biases = self._read_and_reshape_biases(f, layer_sizes)

        except FileNotFoundError:
            print(f"Error: Network file '{filepath}' not found.", file=sys.stderr)
            sys.exit(ERROR_CODE)
        except struct.error as e:
            print(f"Error: Network file structure is corrupted: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)

        return layer_sizes, weights, biases