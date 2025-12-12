/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Layer
*/

#pragma once

#include <string>
#include <cmath>
#include <cstdint>
#include "Matrix.hpp"

namespace my_torch {
    class Layer {
        public:
            Layer(u_int32_t input_size, u_int32_t output_size);
            ~Layer() = default;

            Matrix forward(const Matrix& input);

            Matrix& getWeights() { return weights; }
            Matrix& getBiases() { return biases; }
            Matrix& getCacheInput() { return cache_input; }
            Matrix& getCacheZ() { return cache_z; }

            void setWeights(std::vector<double> weights);
            void setBiases(std::vector<double> biases);

            std::string getActivationType() const { return _activation_type; }

            double activate_derivative(double x);
            Matrix backward(const Matrix& gradient, double learning_rate);

        private:
            Matrix weights;
            Matrix biases;
            std::string _activation_type;
            std::function<double(double)> _act_func;

            uint32_t _input_size;
            uint32_t _output_size;

            Matrix cache_input;
            Matrix cache_z;
            Matrix cache_a;

            double activate(double x);
        };
}
