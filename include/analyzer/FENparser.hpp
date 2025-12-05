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

        int parse(const std::string& fenfile);
        void char_value(const char& c, std::vector<float> &board_value);

    protected:
    private:
        std::vector<std::vector<float>> _input;
};
