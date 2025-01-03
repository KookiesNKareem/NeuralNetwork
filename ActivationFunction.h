#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "Matrix.h"
#include <functional>

class ActivationFunction {
public:
    static Matrix apply(const Matrix& input, const std::function<double(double)>& func);
    static Matrix derivative(const Matrix& input, const std::function<double(double)>& func);
    
    static double relu(double x);
    static double reluDerivative(double x);
    static double leakyrelu(double x, double alpha);
    static double leakyreluDerivative(double x, double alpha);
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    static double tanh(double x);
    static double tanhDerivative(double x);
    static Matrix softmax(const Matrix& input);
    static Matrix softmaxDerivative(const Matrix& softmaxOutput);
};

#endif // ACTIVATION_FUNCTION_H