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
#include "FENparser.hpp"
#include "Analyzer.hpp"

namespace my_torch {
    class Network {
        public:
            Network();
            ~Network();

            void init_network(const std::string loadfile, const std::string& chessfile);

            void parse_untrained_nn(const std::string &blanknn);
            void pack_trained_nn(const std::string& trainfile);

            void predict();
            void train(double learning_rate, const std::string& savefile);

            Matrix forward(Matrix input);
            void backward(Matrix& gradient, double learning_rate);

        protected:
        private:
            const int BATCH_SIZE = 64;
            std::vector<Layer> layers;
            int _layerCount;
            std::vector<uint32_t> _layer_size;
            FENparser _parser;
            Analyzer _analyzer;

            std::vector<my_torch::Matrix> _matrix_input;
            std::vector<my_torch::Matrix> _matrix_output;
    };
}