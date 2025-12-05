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

class Network {
    public:
        Network();
        ~Network();

        void parse_untrained_nn(const std::string &blanknn);

    protected:
    private:
        uint32_t _nb_layer;
        TopologyLayer_t _topology;
        WeightsBiasesLayer_t _weightsBiases;
};
