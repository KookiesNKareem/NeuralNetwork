//
// Created by Kareem Fareed on 1/2/25.
//

#ifndef LAYER_H
#define LAYER_H
#include "Matrix.h"
#include "functional"


class Layer {
    Matrix weights;
    Matrix biases;
    Matrix output;
    Matrix preActivation;
    std::function<double(double)> activationFunc;
    std::function<double(double)> activationDerivative;
public:
    Layer(int inputSize, int outputSize, const std::function<double(double)>& activation, const std::function<double(double)>& derivative);
    [[nodiscard]] Matrix forward(const Matrix& inputs);
    std::tuple<Matrix, Matrix, Matrix> backward(const Matrix& dA, const Matrix& A_prev);
    void update(const Matrix& dW, const Matrix& db, double learningRate);
    void printParameters() const;
    [[nodiscard]] Matrix getWeights() const;
    [[nodiscard]] Matrix getBiases() const;
};



#endif //LAYER_H
