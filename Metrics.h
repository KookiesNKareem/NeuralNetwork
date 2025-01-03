//
// Created by Kareem Fareed on 1/2/25.
//

#ifndef METRICS_H
#define METRICS_H
#include "Matrix.h"


class Metrics {
public:
    static double accuracy(const Matrix& yTrue, const Matrix& yPred);
    static double r2(const Matrix& yTrue, const Matrix& yPred);
    static double meanAbsoluteError(const Matrix& yTrue, const Matrix& yPred);
};



#endif //METRICS_H
