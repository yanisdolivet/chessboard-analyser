/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Network
*/

#include "Network.hpp"

Network::Network()
{
}

Network::~Network()
{
}

void Network::parse_untrained_nn(const std::string &blanknn)
{
    NetwokrHeader_t header;

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

        this->_topology.layerSizes.push_back(nb_wandb);
    }

    for (unsigned int i = 1; i < header.layerCount; i++) {
        uint32_t l1 = this->_topology.layerSizes.at(i-1);
        uint32_t l2 = this->_topology.layerSizes.at(i);
        uint32_t totalweights = l1 * l2;

        for (unsigned int j = 0; j < totalweights; j++) {
            float weight;
            ifd.read(reinterpret_cast<char *>(&weight), sizeof(float));
            this->_weightsBiases.weights.push_back(weight);
        }
    }

    for (unsigned int i = 0; i < header.layerCount; i++) {
        uint32_t nbbiases = this->_topology.layerSizes.at(i);
        for (unsigned int j = 0; j < nbbiases; j++) {
            float biases;
            ifd.read(reinterpret_cast<char *>(&biases), sizeof(float));
            this->_weightsBiases.biases.push_back(biases);
        }
    }
}