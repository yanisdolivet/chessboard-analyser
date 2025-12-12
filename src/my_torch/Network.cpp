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

void my_torch::Network::init_network(const std::string loadfile, const std::string& chessfile)
{
    this->parse_untrained_nn(loadfile); // Parse the nn file
    std::vector<std::vector<double>> inputs = _parser.parse(chessfile, _matrix_output);

    // Transform (each) chess position into a (1, 64) matrix
    _matrix_input = _analyzer.vector_to_matrix(inputs);
}

void my_torch::Network::parse_untrained_nn(const std::string &blanknn)
{
    NetwokrHeader_t header;

    std::ifstream ifd(blanknn, std::ios::binary);
    ifd.seekg(0, std::ios::beg);
    ifd.read(reinterpret_cast<char *>(&header), sizeof(header));

    if (header.magicNumber != MAGIC_NUMBER) {
        std::cerr << "Invalid magic number in FEN file." << std::endl;
        return;
    }

    this->_layerCount = header.layerCount;

    for (unsigned int i = 0; i < header.layerCount; i++) {
        uint32_t nb_wandb;
        ifd.read(reinterpret_cast<char *>(&nb_wandb), sizeof(uint32_t));
        std::cout << "layer size at " << i << " : " << nb_wandb << std::endl;
        _layer_size.push_back(nb_wandb);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t l1 = _layer_size.at(i-1);
        uint32_t l2 = _layer_size.at(i);
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
        uint32_t nbbiases = _layer_size.at(i);
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

void my_torch::Network::pack_trained_nn(const std::string& trainfile)
{
    std::FILE* f = std::fopen(trainfile.c_str(), "wb");

    if (!f) {
        std::cerr << "File couldn't be opened !" << std::endl;
        return;
    }

    // Write header (magic number and layer count)
    uint32_t magic = MAGIC_NUMBER;
    uint32_t layerCount = this->_layerCount;
    std::fwrite(&magic, sizeof(uint32_t), 1, f);
    std::fwrite(&layerCount, sizeof(uint32_t), 1, f);

    std::cout << "Writing " << layerCount << " layers" << std::endl;

    // Write layer sizes (topology)
    for (unsigned int i = 0; i < layerCount; i++) {
        uint32_t layerSize = this->_layer_size.at(i);
        std::cout << "Layer " << i << " size: " << layerSize << std::endl;
        std::fwrite(&layerSize, sizeof(uint32_t), 1, f);
    }

    // Write all weights
    for (unsigned int i = 1; i < layerCount; i++) {
        Matrix weights = this->layers.at(i-1).getWeights();
        for (int j = 0; j < weights.getRows(); j++) {
            for (int k = 0; k < weights.getCols(); k++) {
                float weight = static_cast<float>(weights.at(j, k));
                std::fwrite(&weight, sizeof(float), 1, f);
            }
        }
    }

    // Write all biases
    for (unsigned int i = 1; i < layerCount; i++) {
        Matrix biases = this->layers.at(i-1).getBiases();
        for (int j = 0; j < biases.getRows(); j++) {
            for (int k = 0; k < biases.getCols(); k++) {
                float bias = static_cast<float>(biases.at(j, k));
                std::fwrite(&bias, sizeof(float), 1, f);
            }
        }
    }

    std::fclose(f);
    std::cout << "Trained network saved to " << trainfile << std::endl;
}

my_torch::Matrix my_torch::Network::forward(Matrix input)
{
    Matrix current = input;
    for (auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

void my_torch::Network::train(double learning_rate, const std::string& savefile)
{
    for (int i = 0; i < 50; i++) {
        for (std::size_t i = 0; i < _matrix_input.size(); i++) {
            Matrix output = this->forward(_matrix_input[i]);

            double output_value = output.at(0, 0);
            double expected_value = _matrix_output[i].at(0, 0);

            double error = expected_value - output_value;
            double loss = error * error;

            std::cout << "Loss: " << loss << std::endl;
            Matrix gradient = output - _matrix_output[i];
            this->backward(gradient, learning_rate);
        }
    }
    this->pack_trained_nn(savefile);
}

void my_torch::Network::predict()
{
    Matrix output;
    for (std::size_t i = 0; i < _matrix_input.size(); i++) {
            output = this->forward(_matrix_input[i]);
    }
    const double prediction = output.at(0, 0);
    if (prediction <= 0.5)
        std::cout << "Nothing" << std::endl;
    else if (prediction <= 1.0)
        std::cout << "Check" << std::endl;
}

void my_torch::Network::backward(Matrix& gradient, double learning_rate)
{
    Matrix current_gradient = gradient;

    for (int i = layers.size() - 1; i >= 0; --i) {
        current_gradient = layers[i].backward(current_gradient, learning_rate);
    }
}