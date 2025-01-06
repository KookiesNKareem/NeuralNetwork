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

    const double epsilon = 1e-12; // To prevent log(0)
    double sum = 0.0;

    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            double pred = std::clamp(yPred(i, j), epsilon, 1.0 - epsilon);
            sum -= yTrue(i, j) * std::log(pred) + (1.0 - yTrue(i, j)) * std::log(1.0 - pred);
        }
    }
    return sum / yTrue.getCols();
}

Matrix Loss::crossEntropyDerivative(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for Cross-Entropy derivative.");
    }

    const double epsilon = 1e-12;
    Matrix result(yTrue.getRows(), yTrue.getCols());

    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            double pred = std::clamp(yPred(i, j), epsilon, 1.0 - epsilon);
            result(i, j) = (pred - yTrue(i, j)) / (pred * (1.0 - pred));
        }
    }
    return result;
}