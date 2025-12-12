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

enum TypeMode {
    TRAIN,
    PREDICT
};

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

static bool usage_mode(int argc, char *argv[])
{
    return cmdOptionExists(argv, argv+argc, "-h");
}

static bool train_mode(int argc, char *argv[])
{
    return cmdOptionExists(argv, argv+argc, "--train");
}

static bool predict_mode(int argc, char *argv[])
{
    return cmdOptionExists(argv, argv+argc, "--predict");
}

static bool savedfile_mode(int argc, char *argv[])
{
    return cmdOptionExists(argv, argv+argc, "--save");
}

int print_usage()
{
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

int main(int argc, char *argv[]) {
    my_torch::Network network;
    TypeMode mode;

    std::string loadfile;
    std::string chessfile;
    std::string savefile;

    if (usage_mode(argc, argv)) {
        return print_usage();
    }

    if (predict_mode(argc, argv)) {
        if (argc != 4) {
            std::cerr << "Invalid number argument for predict mode." << std::endl;
            return 1;
        }
        loadfile = argv[2];
        chessfile = argv[3];
        mode = TypeMode::PREDICT;
    } else if (train_mode(argc, argv)) {
        if (savedfile_mode(argc, argv)) {
            if (argc != 6) {
                std::cerr << "Invalid number argument for train mode." << std::endl;
                return 1;
            }
            savefile = argv[3];
            loadfile = argv[4];
            chessfile = argv[5];
        } else if (argc != 4) {
            std::cerr << "Invalid number argument for train mode." << std::endl;
            return 1;
        }
        savefile = "trained_network.nn";
        loadfile = argv[2];
        chessfile = argv[3];
        mode = TypeMode::TRAIN;
    }
    network.init_network(loadfile, chessfile);

    if (mode == TypeMode::TRAIN) {
        network.train(0.01, savefile);
    } else if (mode == TypeMode::PREDICT) {
        network.predict();
    }
}
