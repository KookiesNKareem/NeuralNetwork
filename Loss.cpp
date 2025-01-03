//
// Created by Kareem Fareed on 1/2/25.
//

#include "Loss.h"

double Loss::mse(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for MSE.");
    }

    double sum = 0.0;
    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            double diff = yTrue(i, j) - yPred(i, j);
            sum += diff * diff;
        }
    }
    return sum / (yTrue.getRows() * yTrue.getCols());
}

Matrix Loss::mseDerivative(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for MSE derivative.");
    }

    Matrix result(yTrue.getRows(), yTrue.getCols());
    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            result(i, j) = -2 * (yTrue(i, j) - yPred(i, j)) / (yTrue.getRows() * yTrue.getCols());
        }
    }
    return result;
}

double Loss::crossEntropy(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for Cross-Entropy Loss.");
    }

    double sum = 0.0;
    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            if (yTrue(i, j) > 0) {
                sum -= yTrue(i, j) * std::log(yPred(i, j) + 1e-15); // Add epsilon for stability
            }
        }
    }
    return sum / yTrue.getCols(); // Average loss over batch
}

Matrix Loss::crossEntropyDerivative(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for Cross-Entropy derivative.");
    }

    Matrix result(yTrue.getRows(), yTrue.getCols());
    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            result(i, j) = -yTrue(i, j) / (yPred(i, j) + 1e-15);
        }
    }
    return result;
}