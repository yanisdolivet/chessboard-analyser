/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** FENparser
*/

#include "FENparser.hpp"

FENparser::FENparser()
{
}

FENparser::~FENparser()
{
}

void FENparser::char_value(const char& c, std::vector<double> &board_value)
{
    std::map<char, int> piece_value = {
        {'K', 6}, {'Q', 5}, {'R', 4}, {'B', 3}, {'N', 2}, {'P', 1},
        {'k', -6}, {'q', -5}, {'r', -4}, {'b', -3}, {'n', -2}, {'p', -1},
    };

    if (c == '/') {
        return;
    } else if (c > '0' && c < '9') {
        int nbofzero = c - '0';
        for (int j = 0; j < nbofzero; j++) board_value.push_back((double)0);
    } else {
        board_value.push_back((double)piece_value[c] / (double)6); // Normalisation
    }
}

std::vector<std::vector<double>> FENparser::parse(const std::string& fenfile)
{
    std::ifstream ifd(fenfile);
    std::vector<std::vector<double>> input_data;

    if (!ifd.is_open()) {
        std::cerr << "Error during the opening of the file !" << std::endl;
        return input_data;
    }

    std::string board_content;
    std::string line;
    while(getline(ifd, line)) {
        std::stringstream ss(line);
        getline(ss, board_content, ' ');

        std::vector<double> board_value;
        for(std::string::size_type i = 0; i < board_content.size(); ++i) {
            char_value(board_content[i], board_value);
        }
        input_data.push_back(board_value);
    }
    return input_data;
}

double FENparser::maping_result(const std::string result)
{
    if (result == "Nothing")
        return double(0);
    else if (result == "Check")
        return (double)0.5;
    else if (result == "Checkmate")
        return (double)1;
    return (double)-1;
}
