/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Network
*/

#pragma once

#include <iostream>
#include <fstream>
#include "FileFomat.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

namespace my_torch {
    class Network {
        public:
            Network();
            ~Network();

            void parse_untrained_nn(const std::string &blanknn);
            Matrix forward(Matrix input);

            void train(std::vector<Matrix> inputs, std::vector<Matrix> expected_outputs, double learning_rate);

            void backward(Matrix& gradient, double learning_rate);

        protected:
        private:
            std::vector<Layer> layers;
    };
}