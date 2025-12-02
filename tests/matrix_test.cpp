/*
** EPITECH PROJECT, 2025
** chessboard-analyser
** File description:
** matrix_test
*/

#include <criterion/criterion.h>
#include "../include/my_torch/Matrix.hpp"

Test(MatrixTest, Multiplication) {
    my_torch::Matrix A(2, 3);
    my_torch::Matrix B(3, 2);

    A.at(0, 0) = 1; A.at(0, 1) = 2; A.at(0, 2) = 3;
    A.at(1, 0) = 4; A.at(1, 1) = 5; A.at(1, 2) = 6;

    B.at(0, 0) = 7;  B.at(0, 1) = 8;
    B.at(1, 0) = 9;  B.at(1, 1) = 10;
    B.at(2, 0) = 11; B.at(2, 1) = 12;

    my_torch::Matrix C = A.multiply(B);

    cr_assert_eq(C.getRows(), 2);
    cr_assert_eq(C.getCols(), 2);
    cr_assert_eq(C.at(0, 0), 58);
    cr_assert_eq(C.at(0, 1), 64);
    cr_assert_eq(C.at(1, 0), 139);
    cr_assert_eq(C.at(1, 1), 154);
}

Test(MatrixTest, Transpose) {
    my_torch::Matrix A(2, 3);
    A.at(0, 0) = 1; A.at(0, 1) = 2; A.at(0, 2) = 3;
    A.at(1, 0) = 4; A.at(1, 1) = 5; A.at(1, 2) = 6;

    my_torch::Matrix At = A.transpose();

    cr_assert_eq(At.getRows(), 3);
    cr_assert_eq(At.getCols(), 2);
    cr_assert_eq(At.at(0, 0), 1);
    cr_assert_eq(At.at(0, 1), 4);
    cr_assert_eq(At.at(1, 0), 2);
    cr_assert_eq(At.at(1, 1), 5);
    cr_assert_eq(At.at(2, 0), 3);
    cr_assert_eq(At.at(2, 1), 6);
}

Test(MatrixTest, Addition) {
    my_torch::Matrix A(2, 2);
    my_torch::Matrix B(2, 2);

    A.at(0, 0) = 1; A.at(0, 1) = 2;
    A.at(1, 0) = 3; A.at(1, 1) = 4;

    B.at(0, 0) = 5; B.at(0, 1) = 6;
    B.at(1, 0) = 7; B.at(1, 1) = 8;

    my_torch::Matrix C = A + B;

    cr_assert_eq(C.getRows(), 2);
    cr_assert_eq(C.getCols(), 2);
    cr_assert_eq(C.at(0, 0), 6);
    cr_assert_eq(C.at(0, 1), 8);
    cr_assert_eq(C.at(1, 0), 10);
    cr_assert_eq(C.at(1, 1), 12);
}

Test(MatrixTest, Subtraction) {
    my_torch::Matrix A(2, 2);
    my_torch::Matrix B(2, 2);

    A.at(0, 0) = 5; A.at(0, 1) = 6;
    A.at(1, 0) = 7; A.at(1, 1) = 8;

    B.at(0, 0) = 1; B.at(0, 1) = 2;
    B.at(1, 0) = 3; B.at(1, 1) = 4;

    my_torch::Matrix C = A - B;

    cr_assert_eq(C.getRows(), 2);
    cr_assert_eq(C.getCols(), 2);
    cr_assert_eq(C.at(0, 0), 4);
    cr_assert_eq(C.at(0, 1), 4);
    cr_assert_eq(C.at(1, 0), 4);
    cr_assert_eq(C.at(1, 1), 4);
}

Test(MatrixTest, ScalarMultiplication) {
    my_torch::Matrix A(2, 2);

    A.at(0, 0) = 1; A.at(0, 1) = 2;
    A.at(1, 0) = 3; A.at(1, 1) = 4;

    double scalar = 3.0;
    my_torch::Matrix B = A * scalar;

    cr_assert_eq(B.getRows(), 2);
    cr_assert_eq(B.getCols(), 2);
    cr_assert_eq(B.at(0, 0), 3);
    cr_assert_eq(B.at(0, 1), 6);
    cr_assert_eq(B.at(1, 0), 9);
    cr_assert_eq(B.at(1, 1), 12);
}