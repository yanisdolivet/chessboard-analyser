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
    std::map<char, int> piece_index = {
        {'P', 0}, {'N', 1}, {'B', 2}, {'R', 3}, {'Q', 4}, {'K', 5},
        {'p', 6}, {'n', 7}, {'b', 8}, {'r', 9}, {'q', 10}, {'k', 11}
    };

    if (c == '/') {
        return;
    }

    else if (c >= '1' && c <= '8') {
        int nb_empty_squares = c - '0';
        for (int j = 0; j < nb_empty_squares; j++) {
            for (int k = 0; k < 12; k++) {
                board_value.push_back(0.0);
            }
        }
    }

    else if (piece_index.count(c)) {
        int index = piece_index.at(c);
        for (int k = 0; k < 12; k++) {
            if (k == index) {
                board_value.push_back(1.0);
            } else {
                board_value.push_back(0.0);
            }
        }
    }
}

my_torch::Matrix FENparser::maping_result(const std::string result)
{
    my_torch::Matrix output_matrix(1, 3, false);

    if (result == "Nothing") {
        output_matrix.at(0, 0) = 1.0; // [1, 0, 0]
    } else if (result == "Check") {
        output_matrix.at(0, 1) = 1.0; // [0, 1, 0]
    } else if (result == "Checkmate") {
        output_matrix.at(0, 2) = 1.0; // [0, 0, 1]
    } else {
        std::cerr << "Warning: Unknown result word '" << result << "'. Assuming Nothing." << std::endl;
        output_matrix.at(0, 0) = 1.0;
    }
    return output_matrix;
}

std::vector<std::vector<double>> FENparser::parse(const std::string& fenfile, std::vector<my_torch::Matrix>& output_matrices)
{
    std::ifstream ifd(fenfile);
    std::vector<std::vector<double>> input_data;
    output_matrices.clear();

    if (!ifd.is_open()) {
        std::cerr << "Error during the opening of the file !" << std::endl;
        return input_data;
    }

    std::string line;
    while(getline(ifd, line)) {
        std::stringstream ss(line);
        std::string board_content;
        std::string result_word;

        size_t last_space = line.rfind(' ');
        if (last_space != std::string::npos) {
            board_content = line.substr(0, last_space);
            result_word = line.substr(last_space + 1);
        } else {
            board_content = line;
            result_word = "Nothing"; // Valeur par défaut
        }

        std::stringstream board_ss(board_content);
        std::string board_position;
        getline(board_ss, board_position, ' ');

        std::vector<double> board_value;
        for(std::string::size_type i = 0; i < board_position.size(); ++i) {
            char_value(board_position[i], board_value);
        }

        if (board_value.size() != 768) {
            std::cerr << "Error: FEN input size is " << board_value.size() << ", expected 768." << std::endl;
            continue;
        }

        input_data.push_back(board_value);

        my_torch::Matrix output_matrix = maping_result(result_word);
        output_matrices.push_back(output_matrix);
    }
    return input_data;
}
