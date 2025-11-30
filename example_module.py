"""Example module demonstrating Google-style docstrings.

This module shows how to write documentation that Sphinx can automatically
generate from your code using the Napoleon extension.
"""


def calculate_sum(a, b):
    """Calculate the sum of two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of a and b.

    Example:
        >>> calculate_sum(2, 3)
        5
    """
    return a + b


def divide_numbers(numerator, denominator):
    """Divide two numbers with error handling.

    Args:
        numerator (float): The number to be divided.
        denominator (float): The number to divide by.

    Returns:
        float: The result of the division.

    Raises:
        ValueError: If denominator is zero.

    Example:
        >>> divide_numbers(10, 2)
        5.0
    """
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    return numerator / denominator


class ChessAnalyzer:
    """A class to analyze chess positions.

    This class provides methods to analyze chess board positions
    and calculate various metrics.

    Attributes:
        board_size (int): The size of the chess board (default is 8).
        position (dict): Current position state.
    """

    def __init__(self, board_size=8):
        """Initialize the ChessAnalyzer.

        Args:
            board_size (int, optional): Size of the board. Defaults to 8.
        """
        self.board_size = board_size
        self.position = {}

    def analyze_position(self, position_data):
        """Analyze a chess position.

        Args:
            position_data (dict): Dictionary containing piece positions.
                Format: {'piece': (row, col), ...}

        Returns:
            dict: Analysis results containing:
                - piece_count (int): Number of pieces on board
                - coverage (float): Percentage of board covered

        Raises:
            ValueError: If position_data is invalid.

        Note:
            This is a simplified example. Real analysis would be more complex.
        """
        if not isinstance(position_data, dict):
            raise ValueError("position_data must be a dictionary")

        self.position = position_data
        return {
            'piece_count': len(position_data),
            'coverage': len(position_data) / (self.board_size ** 2) * 100
        }
