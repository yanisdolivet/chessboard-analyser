/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** main
*/

#include <iostream>
#include <algorithm>
#include "Network.hpp"
#include "FENparser.hpp"
#include "Matrix.hpp"
#include "Analyzer.hpp"

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

double f(double n)
{
    return (double)(1 + n);
}

int main(int argc, char *argv[]) {
    my_torch::Network network;
    FENparser parser;
    Analyzer analyzer;

    if (cmdOptionExists(argv, argv+argc, "-h")) {
        std::cout << "USAGE" << std::endl;
        std::cout << "\t./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE" << std::endl;
        std::cout << "DESCRIPTION" << std::endl;
        std::cout << "\t--train Launch the neural network in training mode. Each chessboard in FILE must" << std::endl;
        std::cout << "contain inputs to send to the neural network in FEN notation and the expected output" << std::endl;
        std::cout << "separated by space. If specified, the newly trained neural network will be saved in" << std::endl;
        std::cout << "SAVEFILE. Otherwise, it will be saved in the original LOADFILE." << std::endl;
        std::cout << "\t--predict Launch the neural network in prediction mode. Each chessboard in FILE must" << std::endl;
        std::cout << "contain inputs to send to the neural network in FEN notation, and optionally an expected" << std::endl;
        std::cout << "output." << std::endl;
        std::cout << "\t--save Save neural network into SAVEFILE. Only works in train mode." << std::endl;
        std::cout << "\tLOADFILE File containing an artificial neural network" << std::endl;
        std::cout << "\tCHESSFILE File containing chessboards" << std::endl;
        return 0;
    }

    std::string loadfile = argv[1];
    network.parse_untrained_nn(loadfile); // Parse the nn file

    // Parse the chess position into a vector of double
    std::string chessfile = argv[2];
    std::vector<std::vector<double>> inputs = parser.parse(chessfile);

    // Transform (each) chess position into a (1, 64) matrix
    std::vector<my_torch::Matrix> matrix_input = analyzer.vector_to_matrix(inputs);

    // Forward propagation 1x for each position
    for (my_torch::Matrix pos: matrix_input) {
        my_torch::Matrix res = network.forward(pos);
        res.print();
    }
}
