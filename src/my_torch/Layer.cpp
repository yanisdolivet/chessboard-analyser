/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Layer
*/

#include "Layer.hpp"

double relu(double in)
{
    return (in < 0) ? (double)0: in;
}

double sigmoid(double in) {
    return 1/1+std::exp(-in);
}

double none(double in) {
    return in;
}

my_torch::Layer::Layer(u_int32_t input_size, u_int32_t output_size)
    : _input_size(input_size), _output_size(output_size)
{
    _activation_type = "relu";
    if (_activation_type == "relu") {
        this->_act_func = relu;
    } else if (_activation_type == "sigmoid") {
        this->_act_func = sigmoid;
    } else {
        this->_act_func = none;
    }
}

my_torch::Matrix my_torch::Layer::forward(const Matrix& input)
{
    this->cache_input = input;

    Matrix z = input.multiply(this->weights) + this->biases;
    this->cache_z = z;

    Matrix a = z.apply(this->_act_func);
    this->cache_a = a;
    return a;
}


void my_torch::Layer::setWeights(std::vector<double> weights)
{
    Matrix w(this->_input_size, this->_output_size);
    w.filled(weights);
    this->weights = w;
}

void my_torch::Layer::setBiases(std::vector<double> biases)
{
    Matrix b(1, this->_output_size);
    b.filled(biases);
    this->biases = b;
}

double my_torch::Layer::activate_derivative(double x) {
    if (_activation_type == "relu") {
        return (x > 0) ? 1.0 : 0.0;
    } else if (_activation_type == "sigmoid") {
        double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    }
    return 1.0;
}

my_torch::Matrix my_torch::Layer::backward(const Matrix& gradient, double learning_rate)
{

    Matrix dZ = this->cache_z;

    for (int i = 0; i < dZ.getRows(); i++) {
        for (int j = 0; j < dZ.getCols(); j++) {
            double z_val = this->cache_z.at(i, j);
            double grad_val = gradient.at(i, j);
            double derivative = this->activate_derivative(z_val);

            dZ.at(i, j) = grad_val * derivative;
        }
    }

    Matrix input_T = this->cache_input.transpose();
    Matrix dW = input_T.multiply(dZ);

    Matrix dB = dZ;

    Matrix weights_T = this->weights.transpose();
    Matrix dX_prev = dZ.multiply(weights_T);

    Matrix step_W = dW * learning_rate;
    Matrix step_B = dB * learning_rate;

    this->weights = this->weights - step_W;
    this->biases  = this->biases - step_B;

    return dX_prev;
}