//
// Created by Kareem Fareed on 1/2/25.
//

#include "NeuralNetwork.h"

#include <iostream>

NeuralNetwork::NeuralNetwork(double lr) : learningRate(lr) {}

void NeuralNetwork::addLayer(int inputSize, int outputSize,
                             const std::function<double(double)>& activation,
                             const std::function<double(double)>& activationDerivative) {
    if (layers.empty()) {
        layers.emplace_back(inputSize, outputSize, activation, activationDerivative);
    } else {
        // Ensure inputSize matches the outputSize of the last layer
        int prevOutputSize = layers.back().getWeights().getRows();
        if (inputSize != prevOutputSize) {
            throw std::invalid_argument("Input size of the new layer must match the output size of the previous layer.");
        }
        layers.emplace_back(inputSize, outputSize, activation, activationDerivative);
    }
}

Matrix NeuralNetwork::forward(const Matrix& inputs) {
    Matrix output = inputs;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Matrix& yTrue, const Matrix& yPred, const Matrix& inputs) {
    std::vector<Matrix> activations;
    activations.push_back(inputs);

    // Store intermediate activations
    Matrix current = inputs;
    for (auto& layer : layers) {
        current = layer.forward(current);
        activations.push_back(current);
    }

    Matrix dLoss = Loss::mseDerivative(yTrue, yPred);

    for (int i = layers.size() - 1; i >= 0; --i) {
        auto [dW, db, dA_prev] = layers[i].backward(dLoss, activations[i]);
        layers[i].update(dW, db, learningRate);
        dLoss = dA_prev;
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& y, int epochs, const std::string& lossType) {
    for (int epoch = 0; epoch < epochs; ++epoch) {

        Matrix yPred = forward(X);
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i + 1 << ": "
                      << "Weights Dimensions: " << layers[i].getWeights().getRows() << "x" << layers[i].getWeights().getCols()
                      << ", Biases Dimensions: " << layers[i].getBiases().getRows() << "x" << layers[i].getBiases().getCols() << std::endl;
        }

        double loss;

        if (lossType == "mse") {
            loss = Loss::mse(y, yPred);
        } else if (lossType == "crossentropy") {
            loss = Loss::crossEntropy(y, yPred);
        } else {
            throw std::invalid_argument("Unsupported loss type");
        }

        backward(y, yPred, X);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss << std::endl;
    }
}

Matrix NeuralNetwork::predict(const Matrix& X) {
    return forward(X);
}