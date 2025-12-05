/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** main
*/

#include <iostream>
#include <algorithm>
#include "Network.hpp"

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[]) {
    Network network;

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

    std::string filename = argv[1];
    network.parse_untrained_nn(filename);
}
