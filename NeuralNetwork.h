//
// Created by Kareem Fareed on 1/2/25.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "Matrix.h"
#include "Layer.h"
#include "Loss.h"

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    double learningRate;

public:
    NeuralNetwork(double lr);
    void addLayer(int inputSize, int outputSize,
                  const std::function<double(double)>& activation,
                  const std::function<double(double)>& activationDerivative);
    Matrix forward(const Matrix& inputs);
    void backward(const Matrix& yTrue, const Matrix& yPred, const Matrix& inputs, const std::string& lossType);
    void train(const Matrix& X, const Matrix& y, int epochs, const std::string& lossType = "mse");
    Matrix predict(const Matrix& X);
private:
    std::vector<Matrix> activations;
};



#endif //NEURALNETWORK_H
