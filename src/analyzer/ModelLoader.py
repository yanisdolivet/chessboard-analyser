import numpy as np
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAGIC_NUMBER = 0x48435254
VERSION = 2
ERROR_CODE = 84


class ModelLoader:
    """
    Gère la lecture des fichiers binaires du réseau neuronal (.nn)
    et charge la topologie, les poids et les biais dans des tableaux NumPy.
    """

    def __init__(self):
        pass

    def _read_header(self, f) -> tuple[int, int]:
        """Read and validate the Magic Number, returning layer count and version.
        Args:
            f (file object): Opened binary file object.
        Returns:
            tuple: (layer_count (int), version (int))
        """
        header_data = f.read(12)
        if len(header_data) < 12:
            # Try old format (version 1 - no version field)
            if len(header_data) >= 8:
                f.seek(0)
                header_data = f.read(8)
                magic_number, layer_count = struct.unpack("II", header_data)

                if magic_number != MAGIC_NUMBER:
                    print("Error: Invalid magic number in NN file.", file=sys.stderr)
                    sys.exit(ERROR_CODE)

                return layer_count, 1  # Old format
            else:
                print("Error: Invalid header size or empty file.", file=sys.stderr)
                sys.exit(ERROR_CODE)

        magic_number, version, layer_count = struct.unpack("III", header_data)

        if magic_number != MAGIC_NUMBER:
            print("Error: Invalid magic number in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        if layer_count < 2:
            print(
                "Error: Network must have at least an input and an output layer.",
                file=sys.stderr,
            )
            sys.exit(ERROR_CODE)

        return layer_count, version

    def _read_layer_sizes(self, f, layer_count: int) -> list[int]:
        """Read the size of each layer (the topology).

        Args:
            f (file object): Opened binary file object.
            layer_count (int): Number of layers in the network.

        Returns:
            list[int]: List of layer sizes.
        """
        topology_data = f.read(layer_count * 4)
        if len(topology_data) < layer_count * 4:
            print("Error: Missing layer size information.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        layer_sizes = list(struct.unpack(f"{layer_count}I", topology_data))
        return layer_sizes

    def _read_and_reshape_weights(self, f, layer_sizes: list[int]) -> list[np.ndarray]:
        """Read the weights segment, deserialize and reshape into matrices.
        Args:
            f (file object): Opened binary file object.
            layer_sizes (list[int]): List of layer sizes.
        Returns:
            list[np.ndarray]: List of weight matrices for each layer.
        """
        weights = []
        layer_count = len(layer_sizes)

        total_weights = sum(
            layer_sizes[i - 1] * layer_sizes[i] for i in range(1, layer_count)
        )

        weight_data = f.read(total_weights * 4)
        if len(weight_data) < total_weights * 4:
            print("Error: Incomplete weight data in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        flat_weights = struct.unpack(f"{total_weights}f", weight_data)

        current_idx = 0
        for i in range(1, layer_count):
            input_size = layer_sizes[i - 1]
            output_size = layer_sizes[i]
            matrix_size = input_size * output_size

            w_segment = flat_weights[current_idx : current_idx + matrix_size]

            w_matrix = np.array(w_segment, dtype=np.float32).reshape(
                (input_size, output_size)
            )

            weights.append(w_matrix)
            current_idx += matrix_size

        return weights

    def _read_and_reshape_biases(self, f, layer_sizes: list[int]) -> list[np.ndarray]:
        """Read the biases segment, deserialize and reshape into row vectors.

        Args:
            f (file object): Opened binary file object.
            layer_sizes (list[int]): List of layer sizes.

        Returns:
            list[np.ndarray]: List of bias vectors for each layer.
        """
        biases = []
        layer_count = len(layer_sizes)

        total_biases = sum(layer_sizes[1:])

        bias_data = f.read(total_biases * 4)
        if len(bias_data) < total_biases * 4:
            print("Error: Incomplete bias data in NN file.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        flat_biases = struct.unpack(f"{total_biases}f", bias_data)

        current_idx = 0
        for i in range(1, layer_count):
            output_size = layer_sizes[i]

            b_segment = flat_biases[current_idx : current_idx + output_size]

            b_vector = np.array(b_segment, dtype=np.float32).reshape((1, output_size))

            biases.append(b_vector)
            current_idx += output_size

        return biases

    def _decode_string(self, f):
        """Decode length-prefixed string.
        Args:
            f (file object): Opened binary file object.
        Returns:
            str: Decoded string.
        """
        length_data = f.read(4)
        if len(length_data) < 4:
            raise struct.error("Cannot read string length")
        length = struct.unpack("I", length_data)[0]
        string_data = f.read(length)
        if len(string_data) < length:
            raise struct.error("Cannot read string data")
        return string_data.decode("utf-8")

    def _read_configuration(self, f, layer_count):
        """Read network configuration from binary file (version 2+).
        Args:
            f (file object): Opened binary file object.
            layer_count (int): Number of layers in the network.
        Returns:
            ModelSpecifications: Object containing network configuration.
        """
        from src.model_specification import ModelSpecifications

        spec = ModelSpecifications()
        spec.num_layers = layer_count

        # Read layer types and activations
        for i in range(layer_count):
            layer_type = self._decode_string(f)
            activation = self._decode_string(f)
            spec.type.append(layer_type)
            spec.activation.append(activation)

        # Read hyperparameters
        lr_data = f.read(4)
        spec.learning_rate = struct.unpack("f", lr_data)[0]
        spec.initialization = self._decode_string(f)

        # Read training parameters
        batch_data = f.read(4)
        spec.batch_size = struct.unpack("I", batch_data)[0]

        epochs_data = f.read(4)
        spec.epochs = struct.unpack("I", epochs_data)[0]

        lreg_data = f.read(4)
        spec.lreg = struct.unpack("f", lreg_data)[0]

        dropout_data = f.read(4)
        spec.dropout_rate = struct.unpack("f", dropout_data)[0]

        spec.loss_function = self._decode_string(f)

        return spec

    def load_network(self, filepath: str) -> tuple[list, list, list, object]:
        """Load neural network from binary file.
        Args:
            filepath (str): Path to the .nn file.
        Returns:
            tuple: (layer_sizes, weights, biases, model_specification)
        """

        try:
            with open(filepath, "rb") as f:

                layer_count, version = self._read_header(f)
                layer_sizes = self._read_layer_sizes(f, layer_count)

                # Read configuration if version 2+, otherwise create default
                if version >= 2:
                    model_spec = self._read_configuration(f, layer_count)
                    model_spec.layer_sizes = layer_sizes
                else:
                    # Create minimal spec for old format
                    from src.model_specification import ModelSpecifications

                    model_spec = ModelSpecifications()
                    model_spec.num_layers = layer_count
                    model_spec.layer_sizes = layer_sizes
                    model_spec.type = ["HIDDEN"] * layer_count
                    model_spec.activation = ["none"] * layer_count
                    model_spec.learning_rate = 0.01
                    model_spec.initialization = "unknown"
                    model_spec.batch_size = 32
                    model_spec.epochs = 100
                    model_spec.lreg = 0.0
                    model_spec.dropout_rate = 0.0
                    model_spec.loss_function = "mse"

                weights = self._read_and_reshape_weights(f, layer_sizes)
                biases = self._read_and_reshape_biases(f, layer_sizes)

        except FileNotFoundError:
            print(f"Error: Network file '{filepath}' not found.", file=sys.stderr)
            sys.exit(ERROR_CODE)
        except struct.error as e:
            print(f"Error: Network file structure is corrupted: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)

        return layer_sizes, weights, biases, model_spec
