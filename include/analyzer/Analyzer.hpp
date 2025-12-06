/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Analyzer
*/

#pragma once

#include "Matrix.hpp"

#define INPUTS 64

class Analyzer {
    public:
        Analyzer();
        ~Analyzer();

        std::vector<my_torch::Matrix> vector_to_matrix(const std::vector<std::vector<double>>& input);

    protected:
    private:
};
