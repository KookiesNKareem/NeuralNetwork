//
// Created by Kareem Fareed on 1/2/25.
//

#ifndef MATRIX_H
#define MATRIX_H
#include "vector"
#include "functional"

class Matrix {
    std::vector<std::vector<double>> data;
    int rows, cols;
public:
    Matrix();
    Matrix(int rows, int cols);
    static Matrix random(int rows, int cols, double min = -1.0, double max = 1.0);
    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);

    int getRows() const;
    int getCols() const;
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix dot(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator/(const Matrix& other) const;

    Matrix transpose() const;
    Matrix applyFunction(const std::function<double(double)>& func) const;
    Matrix sumRows() const;

    void print() const;
    static Matrix oneHotEncode(const std::vector<int>& labels, int numClasses);
    static Matrix normalize(const Matrix& data);
    static Matrix standardize(const Matrix& data);

    double mean() const;

    std::vector<double>& operator[](size_t row) {
        if (row >= rows) {
            throw std::out_of_range("Row index out of bounds.");
        }
        return data[row];
    }

    const std::vector<double>& operator[](size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("Row index out of bounds.");
        }
        return data[row];
    }
};



#endif //MATRIX_H
