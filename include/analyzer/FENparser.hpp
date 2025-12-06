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

class FENparser {
    public:
        FENparser();
        ~FENparser();

        std::vector<std::vector<double>> parse(const std::string& fenfile);
        void char_value(const char& c, std::vector<double> &board_value);

        double maping_result(const std::string result);

    protected:
    private:
};
