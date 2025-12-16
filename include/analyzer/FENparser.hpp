/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** FENparser
*/

#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include "Matrix.hpp"

class FENparser {
    public:
        FENparser();
        ~FENparser();

        void char_value(const char& c, std::vector<double> &board_value);

        my_torch::Matrix maping_result(const std::string result);
        std::vector<std::vector<double>> parse(const std::string& fenfile, std::vector<my_torch::Matrix>& output_matrices);

    protected:
    private:
};
