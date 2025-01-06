//
// Created by Kareem Fareed on 1/2/25.
//

#include "Layer.h"
#include <cmath>
#include <iostream>

#include "stdexcept"

Layer::Layer(int inputSize, int outputSize, const std::function<double(double)>& activation, const std::function<double(double)>& derivative):
    weights(Matrix::random(outputSize, inputSize, -1.0, 1.0)),
    biases(Matrix::zeros(outputSize, 1)) {
    // He initialization for ReLU activations
    double scalingFactor = std::sqrt(2.0 / inputSize);
    weights = weights * scalingFactor;
    this->activationFunc = activation;
    this->activationDerivative = derivative;
}

Matrix Layer::forward(const Matrix& inputs) {
    if (inputs.getRows() != weights.getCols()) {
        throw std::invalid_argument("Input dimensions do not match layer weights.");
    }

    preActivation = weights * inputs; // Store Z for backward pass

    // Add bias with broadcasting
    for (int i = 0; i < preActivation.getRows(); ++i) {
        for (int j = 0; j < preActivation.getCols(); ++j) {
            preActivation(i, j) += biases(i, 0);
        }
    }
    // Apply activation
    output = preActivation.applyFunction(activationFunc);
    return output;
}

std::tuple<Matrix, Matrix, Matrix> Layer::backward(const Matrix& dA, const Matrix& A_prev) {
    if (dA.getRows() != output.getRows() || dA.getCols() != output.getCols()) {
        throw std::invalid_argument("Gradient dimensions do not match layer output.");
    }

    // Element-wise multiplication of dA with activation derivative
    Matrix dZ = Matrix(dA.getRows(), dA.getCols());
    for (int i = 0; i < dA.getRows(); ++i) {
        for (int j = 0; j < dA.getCols(); ++j) {
            dZ(i, j) = dA(i, j) * activationDerivative(preActivation(i, j));
        }
    }

    // Compute gradients
    Matrix dW = dZ * A_prev.transpose();
    Matrix db = Matrix(biases.getRows(), 1);
    // Sum across samples for bias gradients
    for (int i = 0; i < dZ.getRows(); ++i) {
        double sum = 0;
        for (int j = 0; j < dZ.getCols(); ++j) {
            sum += dZ(i, j);
        }
        db(i, 0) = sum / dZ.getCols();
    }

    Matrix dA_prev = weights.transpose() * dZ;
    return {dW, db, dA_prev};
}

void Layer::update(const Matrix& dW, const Matrix& db, double learningRate) {
    weights = weights - (dW * learningRate);
    biases = biases - (db * learningRate);
}

void Layer::printParameters() const {
    std::cout << "Weights:\n";
    weights.print();

    std::cout << "\nBiases:\n";
    biases.print();
}

Matrix Layer::getWeights() const {
    return weights;
}

Matrix Layer::getBiases() const {
    return biases;
}