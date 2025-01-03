//
// Created by Kareem Fareed on 1/2/25.
//

#include "ActivationFunction.h"
#include <cmath>

Matrix ActivationFunction::apply(const Matrix& input, const std::function<double(double)>& func) {
    return input.applyFunction(func);
}

Matrix ActivationFunction::derivative(const Matrix& input, const std::function<double(double)>& func) {
    return input.applyFunction(func);
}

double ActivationFunction::relu(double x) {
    return std::max(0.0, x);
}

double ActivationFunction::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double ActivationFunction::leakyrelu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double ActivationFunction::leakyreluDerivative(double x, double alpha = 0.01) {
    return x > 0 ? 1.0 : alpha;
}

double ActivationFunction::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double ActivationFunction::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double ActivationFunction::tanh(double x) {
    return std::tanh(x);
}

double ActivationFunction::tanhDerivative(double x) {
    double t = std::tanh(x);
    return 1 - t * t;
}

Matrix ActivationFunction::softmax(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int j = 0; j < input.getCols(); ++j) {
        // Step 1: Compute the max value in the column
        double maxVal = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < input.getRows(); ++i) {
            maxVal = std::max(maxVal, input(i, j));
        }

        // Step 2: Compute exp(x - maxVal) for each element in the column
        double sum = 0.0;
        for (int i = 0; i < input.getRows(); ++i) {
            result(i, j) = std::exp(input(i, j) - maxVal);
            sum += result(i, j);
        }

        // Step 3: Normalize by the sum of exponentials
        for (int i = 0; i < input.getRows(); ++i) {
            result(i, j) /= sum;
        }
    }

    return result;
}

Matrix ActivationFunction::softmaxDerivative(const Matrix& softmaxOutput) {
    int rows = softmaxOutput.getRows(); // Number of classes
    int cols = softmaxOutput.getCols(); // Batch size

    Matrix result(rows, rows); // Jacobian

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < rows; ++k) {
                if (i == k) {
                    result(i, k) = softmaxOutput(i, j) * (1 - softmaxOutput(i, j));
                } else {
                    result(i, k) = -softmaxOutput(i, j) * softmaxOutput(k, j);
                }
            }
        }
    }

    return result;
}