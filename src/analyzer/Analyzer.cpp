/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Analyzer
*/

#include "Analyzer.hpp"

Analyzer::Analyzer()
{
}

Analyzer::~Analyzer()
{
}

std::vector<my_torch::Matrix> Analyzer::vector_to_matrix(const std::vector<std::vector<double>>& input)
{
    std::vector<my_torch::Matrix> matrix_input;
    for (std::vector<double> game: input) {
        int rows = 1;
        int cols = INPUTS;
        my_torch::Matrix matrix(rows, cols);
        matrix.filled(game);
        matrix_input.push_back(matrix);
    }
    return matrix_input;
}
