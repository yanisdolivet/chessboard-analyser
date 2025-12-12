/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Network
*/

#include "Network.hpp"

my_torch::Network::Network()
{
}

my_torch::Network::~Network()
{
}

void my_torch::Network::parse_untrained_nn(const std::string &blanknn)
{
    NetwokrHeader_t header;
    std::vector<uint32_t> layer_size;

    std::ifstream ifd(blanknn, std::ios::binary);
    ifd.seekg(0, std::ios::beg);
    ifd.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (header.magicNumber != MAGIC_NUMBER) {
        std::cerr << "Invalid magic number in FEN file." << std::endl;
        return;
    }

    for (unsigned int i = 0; i < header.layerCount; i++) {
        uint32_t nb_wandb;
        ifd.read(reinterpret_cast<char *>(&nb_wandb), sizeof(uint32_t));
        layer_size.push_back(nb_wandb);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t l1 = layer_size.at(i-1);
        uint32_t l2 = layer_size.at(i);
        uint32_t totalweights = l1 * l2;
        Layer layer(l1, l2);
        std::vector<double> weights;

        for (unsigned int j = 0; j < totalweights; j++) {
            float weight;
            ifd.read(reinterpret_cast<char *>(&weight), sizeof(float));
            weights.push_back(weight);
        }
        layer.setWeights(weights);
        this->layers.push_back(layer);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t nbbiases = layer_size.at(i);
        std::vector<double> biases;
        for (unsigned int j = 0; j < nbbiases; j++) {
            float biase;
            ifd.read(reinterpret_cast<char *>(&biase), sizeof(float));
            biases.push_back(biase);
        }
        this->layers.at(i-1).setBiases(biases);
    }

    std::cout << "CONFIG FILE PARSE SUCCESSFULLY" << std::endl;
}

my_torch::Matrix my_torch::Network::forward(Matrix input)
{
    Matrix current = input;
    for (auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

void my_torch::Network::train(std::vector<Matrix> inputs, std::vector<Matrix> expected_outputs, double learning_rate)
{
    for (std::size_t i = 0; i < inputs.size(); i++) {
        Matrix output = this->forward(inputs[i]);

        double output_value = output.at(0, 0);
        double expected_value = expected_outputs[i].at(0, 0);

        double error = expected_value - output_value;
        double loss = error * error;

        std::cout << "Loss: " << loss << std::endl;
        Matrix gradient = output - expected_outputs[i];
        this->backward(gradient, learning_rate);
    }
}

void my_torch::Network::backward(Matrix& gradient, double learning_rate)
{
    Matrix current_gradient = gradient;

    for (int i = layers.size() - 1; i >= 0; --i) {
        current_gradient = layers[i].backward(current_gradient, learning_rate);
    }
}