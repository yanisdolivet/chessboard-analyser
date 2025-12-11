/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Matrix
*/

#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <cmath>

namespace my_torch {

    class Matrix {
    public:
        Matrix();
        Matrix(int rows, int cols, bool is_random = false);
        ~Matrix() = default;

        Matrix multiply(const Matrix& other) const;
        Matrix transpose() const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        void filled(std::vector<double> fulldata);

        my_torch::Matrix apply(std::function<double(double)> func);

        void print() const;
        int getRows() const { return rows; }
        int getCols() const { return cols; }
        double& at(int r, int c) { return data[r][c]; }
        const double& at(int r, int c) const { return data[r][c]; }

        void printMatrix() const;

    private:
        int rows;
        int cols;
        std::vector<std::vector<double>> data;
    };

}
