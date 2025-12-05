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

void FENparser::char_value(const char& c, std::vector<float> &board_value)
{
    std::map<char, int> piece_value = {
        {'K', 6}, {'Q', 5}, {'R', 4}, {'B', 3}, {'N', 2}, {'P', 1},
        {'k', -6}, {'q', -5}, {'r', -4}, {'b', -3}, {'n', -2}, {'p', -1},
    };

    if (c == '/') {
        return;
    } else if (c > '0' && c < '9') {
        int nbofzero = c - '0';
        for (int j = 0; j < nbofzero; j++) board_value.push_back((float)0);
    } else {
        board_value.push_back((float)piece_value[c] / (float)6); // Normalisation
    }
}

int FENparser::parse(const std::string& fenfile)
{
    std::ifstream ifd(fenfile);

    if (!ifd.is_open()) {
        std::cerr << "Error during the opening of the file !" << std::endl;
        return 1;
    }

    std::string board_content;
    std::string line;
    while(getline(ifd, line)) {
        std::stringstream ss(line);
        getline(ss, board_content, ' ');

        std::vector<float> board_value;
        for(std::string::size_type i = 0; i < board_content.size(); ++i) {
            char_value(board_content[i], board_value);
        }
        this->_input.push_back(board_value);
    }
    return 0;
}

float FENparser::maping_result(const std::string result)
{
    if (result == "Nothing")
        return float(0);
    else if (result == "Check")
        return (float)0.5;
    else if (result == "Checkmate")
        return (float)1;
    return (float)-1;
}
