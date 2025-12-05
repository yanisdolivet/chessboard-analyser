/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** FileFomat
*/

#ifndef FILEFOMAT_HPP_
    #define FILEFOMAT_HPP_

    #include <cstdint>
    #include <vector>

const uint32_t MAGIC_NUMBER = 0x48435254; // 'TRCH' in little-endian

// Header
typedef struct NetwokrHeader_s {
    uint32_t magicNumber;
    uint32_t layerCount;
} NetwokrHeader_t;

// Topology Layer
typedef struct TopologyLayer_s {
    std::vector<uint32_t> layerSizes;
} TopologyLayer_t;

// Weights and Biases Layer
typedef struct WeightsBiasesLayer_s {
    std::vector<float> weights;
    std::vector<float> biases;
} WeightsBiasesLayer_t;


#endif /* !FILEFOMAT_HPP_ */
