//
// Created by Kareem Fareed on 1/2/25.
//

#include "Matrix.h"

#include <iostream>
#include <random>
#include <stdexcept>

Matrix::Matrix() : rows(0), cols(0) {}
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
  data = std::vector(rows, std::vector(cols, 0.0));
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
  Matrix result(rows, cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
        {
          result(i, j) = dis(gen);
        }
    }
  return result;
}

Matrix Matrix::ones(int rows, int cols) {
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result(i, j) = 1.0;
    }
  }
  return result;
}

Matrix Matrix::zeros(int rows, int cols) {
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result(i, j) = 0.0;
    }
  }
  return result;
}

int Matrix::getRows() const {
  return this->rows;
}

int Matrix::getCols() const {
  return this->cols;
}

double& Matrix::operator()(int row, int col) {
  if (row < 0 || row >= this->rows || col < 0 || col >= this->cols) {
    throw std::out_of_range("Matrix index out of range");
  }
  return this->data[row][col];
}

const double& Matrix::operator()(int row, int col) const {
  if (row < 0 || row >= this->rows || col < 0 || col >= this->cols) {
    throw std::out_of_range("Matrix index out of range");
  }
  return this->data[row][col];
}

Matrix Matrix::operator+(const Matrix& other) const {
  if (this->cols != other.getCols() || this->rows != other.getRows()) {
    throw std::out_of_range("Matrix dimensions do not match");
  }
  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result(i, j) = this->data[i][j] + other.data[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  if (this->cols != other.getCols() || this->rows != other.getRows()) {
    throw std::out_of_range("Matrix dimensions do not match");
  }
  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result(i, j) = this->data[i][j] - other.data[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  if (this->cols != other.getRows()) {
    throw std::out_of_range("Improper dimensions for matrix multiplication");
  }
  Matrix result(this->rows, other.getCols());
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < other.getCols(); j++) {
      double sum = 0.0;
      for (int k = 0; k < this->cols; k++) {
        sum += this->data[i][k] * other.data[k][j];
      }
      result(i, j) = sum;
    }
  }
  return result;
}

// Same function as the * overload, just renamed for convenience/clarity
Matrix Matrix::dot(const Matrix& other) const {
  if (cols != other.getRows()) {
    throw std::out_of_range("Improper dimensions for matrix multiplication");
  }
  Matrix result(rows, other.getCols());
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < other.getCols(); ++j) {
      double sum = 0.0;
      for (int k = 0; k < cols; ++k) {
        sum += data[i][k] * other.data[k][j];
      }
      result(i, j) = sum;
    }
  }
  return result;
}

Matrix Matrix::operator*(double scalar) const {
  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result.data[i][j] = this->data[i][j] * scalar;
    }
  }
  return result;
}

Matrix Matrix::operator/(double scalar) const {
  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result.data[i][j] = this->data[i][j] / scalar;
    }
  }
  return result;
}

void Matrix::print() const {
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      std::cout << this->data[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::transpose() const {
  Matrix result(cols, rows);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result(j, i) = data[i][j];
    }
  }
  return result;
}

Matrix Matrix::sumRows() const {
  Matrix result(rows, 1);
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += data[i][j];
    }
    result(i, 0) = sum;
  }
  return result;
}

Matrix Matrix::applyFunction(const std::function<double(double)>& func) const {
  Matrix result(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result(i, j) = func(data[i][j]);
    }
  }
  return result;
}

Matrix Matrix::oneHotEncode(const std::vector<int>& labels, int numClasses) {
    Matrix result(numClasses, labels.size());

    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || labels[i] >= numClasses) {
            throw std::out_of_range("Label value out of range for one-hot encoding");
        }
        result(labels[i], i) = 1.0;
    }

    return result;
}

// Min-max
Matrix Matrix::normalize(const Matrix& data) {
    Matrix result = data;

    for (int j = 0; j < data.getCols(); ++j) {
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();

        // Find min and max for each column
        for (int i = 0; i < data.getRows(); ++i) {
            minVal = std::min(minVal, data(i, j));
            maxVal = std::max(maxVal, data(i, j));
        }

        // Apply normalization
        for (int i = 0; i < data.getRows(); ++i) {
            if (maxVal > minVal) {
                result(i, j) = (data(i, j) - minVal) / (maxVal - minVal);
            } else {
                result(i, j) = 0.0;
            }
        }
    }

    return result;
}

// Z-score normalization
Matrix Matrix::standardize(const Matrix& data) {
    Matrix result = data;

    for (int j = 0; j < data.getCols(); ++j) {
        double mean = 0.0;
        double stdDev = 0.0;

        for (int i = 0; i < data.getRows(); ++i) {
            mean += data(i, j);
        }
        mean /= data.getRows();

        for (int i = 0; i < data.getRows(); ++i) {
            stdDev += std::pow(data(i, j) - mean, 2);
        }
        stdDev = std::sqrt(stdDev / data.getRows());

        // Apply standardization
        for (int i = 0; i < data.getRows(); ++i) {
            if (stdDev > 0) {
                result(i, j) = (data(i, j) - mean) / stdDev;
            } else {
                result(i, j) = 0.0;
            }
        }
    }

    return result;
}

Matrix Matrix::operator/(const Matrix& other) const {
  if (this->rows != other.getRows() || this->cols != other.getCols()) {
    throw std::invalid_argument("Matrix dimensions must match for element-wise division.");
  }

  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      if (other(i, j) == 0) {
        throw std::runtime_error("Division by zero in element-wise division.");
      }
      result(i, j) = this->data[i][j] / other.data[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator-(double scalar) const {
  Matrix result(this->rows, this->cols);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      result.data[i][j] = this->data[i][j] - scalar;
    }
  }
  return result;
}

double Matrix::mean() const{
  double sum = 0.0;
  size_t totalElements = rows * cols;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      sum += data[i][j];
    }
  }
  if (totalElements == 0) {
    throw std::runtime_error("Matrix has no elements, cannot compute mean.");
  }
  return sum / totalElements;
}
