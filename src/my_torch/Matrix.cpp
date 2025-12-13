/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** Matrix
*/

#include "Matrix.hpp"

my_torch::Matrix::Matrix()
    : rows(0), cols(0), data()
{
}

my_torch::Matrix::Matrix(int rows, int cols, bool is_random)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0))
{
    if (is_random) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                data[r][c] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }
}

my_torch::Matrix my_torch::Matrix::multiply(const Matrix& other) const
{
    if (cols != other.rows) {
        throw std::invalid_argument("Incompatible matrix sizes for multiplication");
    }
    Matrix result(rows, other.cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < other.cols; ++c) {
            for (int k = 0; k < cols; ++k) {
                result.at(r, c) += at(r, k) * other.at(k, c);
            }
        }
    }
    return result;
}

my_torch::Matrix my_torch::Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.at(c, r) = at(r, c);
        }
    }
    return result;
}

my_torch::Matrix my_torch::Matrix::operator+(const Matrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Incompatible matrix sizes for addition");
    }
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.at(r, c) = at(r, c) + other.at(r, c);
        }
    }
    return result;
}

my_torch::Matrix my_torch::Matrix::operator-(const Matrix& other) const
{
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Incompatible matrix sizes for subtraction");
    }
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.at(r, c) = at(r, c) - other.at(r, c);
        }
    }
    return result;
}

my_torch::Matrix my_torch::Matrix::operator*(double scalar) const
{
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.at(r, c) = at(r, c) * scalar;
        }
    }
    return result;
}

void my_torch::Matrix::filled(std::vector<double> fulldata)
{
    int k = 0;

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            this->data[i][j] = fulldata.at(k);
            k++;
        }
    }
}

my_torch::Matrix my_torch::Matrix::apply(std::function<double(double)> func)
{
    Matrix result(rows, cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.at(i, j) = func(data[i][j]);
        }
    }
    return result;
}

void my_torch::Matrix::print() const
{
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            std::cout << this->data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void my_torch::Matrix::printMatrix() const
{
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            std::cout << this->data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}