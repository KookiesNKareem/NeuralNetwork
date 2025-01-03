//
// Created by Kareem Fareed on 1/2/25.
//

#include "Metrics.h"

double Metrics::accuracy(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for accuracy.");
    }

    int correct = 0;
    int total = yTrue.getCols(); // Assume one-hot encoded

    for (int j = 0; j < yTrue.getCols(); ++j) {
        int trueClass = 0, predClass = 0;
        double maxTrue = -1e15, maxPred = -1e15;

        for (int i = 0; i < yTrue.getRows(); ++i) {
            if (yTrue(i, j) > maxTrue) {
                maxTrue = yTrue(i, j);
                trueClass = i;
            }
            if (yPred(i, j) > maxPred) {
                maxPred = yPred(i, j);
                predClass = i;
            }
        }

        if (trueClass == predClass) {
            correct++;
        }
    }

    return static_cast<double>(correct) / total;
}

double Metrics::r2(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for R².");
    }

    double mean = 0.0, ssTotal = 0.0, ssResidual = 0.0;

    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            mean += yTrue(i, j);
        }
    }
    mean /= (yTrue.getRows() * yTrue.getCols());

    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            double diffTrue = yTrue(i, j) - mean;
            double diffResidual = yTrue(i, j) - yPred(i, j);
            ssTotal += diffTrue * diffTrue;
            ssResidual += diffResidual * diffResidual;
        }
    }

    return 1.0 - (ssResidual / ssTotal);
}

double Metrics::meanAbsoluteError(const Matrix& yTrue, const Matrix& yPred) {
    if (yTrue.getRows() != yPred.getRows() || yTrue.getCols() != yPred.getCols()) {
        throw std::invalid_argument("Dimensions of yTrue and yPred must match for MAE.");
    }

    double sum = 0.0;
    for (int i = 0; i < yTrue.getRows(); ++i) {
        for (int j = 0; j < yTrue.getCols(); ++j) {
            sum += std::abs(yTrue(i, j) - yPred(i, j));
        }
    }

    return sum / (yTrue.getRows() * yTrue.getCols());
}