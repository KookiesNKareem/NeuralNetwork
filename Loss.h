//
// Created by Kareem Fareed on 1/2/25.
//

#ifndef LOSS_H
#define LOSS_H
#include "Matrix.h"


class Loss {
public:
    static double mse(const Matrix& yTrue, const Matrix& yPred);
    static Matrix mseDerivative(const Matrix& yTrue, const Matrix& yPred);

    static double crossEntropy(const Matrix& yTrue, const Matrix& yPred);
    static Matrix crossEntropyDerivative(const Matrix& yTrue, const Matrix& yPred);
};



#endif //LOSS_H
