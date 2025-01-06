#include <iostream>

#include "ActivationFunction.h"
#include "Matrix.h"
#include "Layer.h"
#include "Metrics.h"
#include "NeuralNetwork.h"

// void PropagationTest() {
//     // Step 1: Initialize a layer with 2 inputs and 3 outputs
//     Layer layer(2, 3, ActivationFunction::relu, ActivationFunction::reluDerivative);
//     // Step 2: Create random input matrix (2 inputs, 4 samples)
//     Matrix inputs = Matrix::random(2, 4, -1.0, 1.0);
//
//     // Step 3: Perform forward propagation
//     Matrix outputs = layer.forward(inputs);
//     std::cout << "Forward Propagation Output:\n";
//     outputs.print();
//
//     // Step 4: Simulate a gradient of loss w.r.t. outputs (random for testing)
//     Matrix dOutputs = Matrix::random(3, 4, -0.5, 0.5); // Same shape as outputs
//
//     // Step 5: Perform backpropagation
//     auto [dW, db, dA_prev] = layer.backward(dOutputs, inputs);
//     std::cout << "Weight Gradient (dW):\n";
//     dW.print();
//     std::cout << "Bias Gradient (db):\n";
//     db.print();
//     std::cout << "Previous Layer Gradient (dA_prev):\n";
//     dA_prev.print();
//
//     // Step 6: Update weights and biases
//     double learningRate = 0.01;
//     layer.update(dW, db, learningRate);
//
//     // Step 7: Verify parameter updates
//     std::cout << "Updated Weights and Biases:\n";
//     layer.printParameters();
// }
// void TestLayer() {
//     // Initialize a layer with 2 inputs and 3 outputs
//     Layer layer(2, 3, ActivationFunction::relu, ActivationFunction::reluDerivative);
//
//     // Create random inputs: 2 features, 4 samples
//     Matrix inputs = Matrix::random(2, 4, -1.0, 1.0);
//     // Perform forward propagation
//     Matrix outputs = layer.forward(inputs);
//     std::cout << "Forward Propagation Output:\n";
//     outputs.print();
//
//     // Simulate a gradient of the loss w.r.t. the outputs
//     Matrix dOutputs = Matrix::random(3, 4, -0.5, 0.5);
//
//     // Perform backward propagation
//     auto [dW, db, dA_prev] = layer.backward(dOutputs, inputs);
//
//     // Update weights and biases
//     double learningRate = 0.01;
//     layer.update(dW, db, learningRate);
//
//     // Print updated parameters
//     std::cout << "Updated Layer Parameters:\n";
//     layer.printParameters();
// }

void generateSyntheticData(Matrix& X, Matrix& y) {
    int numSamples = 100;

    X = Matrix::random(2, numSamples, -1.0, 1.0);

    y = Matrix(1, numSamples);
    for (int i = 0; i < numSamples; ++i) {
        double x1 = X(0, i);
        double x2 = X(1, i);

        y(0, i) = std::sin(x1) + std::cos(x2);
    }
}

void generateComplexDataset(Matrix& X, Matrix& y) {
    int numSamples = 100;

    // Generate 2D input data (features)
    X = Matrix::random(2, numSamples, -1.0, 1.0);

    // Generate labels based on logical conditions
    y = Matrix(1, numSamples);
    for (int i = 0; i < numSamples; ++i) {
        double x1 = X(0, i);
        double x2 = X(1, i);

        // Classification rule: simple but meaningful
        y(0, i) = (x1 * x1 + x2 * x2 > 0.5) ? 1.0 : 0.0;
    }
}

int main() {
    // Step 1: Initialize the neural network
    std::cout << "Initializing Neural Network..." << std::endl;
    NeuralNetwork nn(0.01);
    nn.addLayer(2, 32, ActivationFunction::relu, ActivationFunction::reluDerivative);
    nn.addLayer(32, 32, ActivationFunction::relu, ActivationFunction::reluDerivative);
    nn.addLayer(32, 32, ActivationFunction::relu, ActivationFunction::reluDerivative);
    nn.addLayer(32, 1, ActivationFunction::sigmoid, ActivationFunction::sigmoidDerivative);

    // Step 2: Generate a simple binary classification dataset
    std::cout << "Generating simple dataset..." << std::endl;
    Matrix X, y;
    generateSyntheticData(X, y);

    // Step 3: Normalize data
    std::cout << "Normalizing data..." << std::endl;
    X = Matrix::normalize(X); // Normalize features

    // Step 4: Train the Neural Network
    std::cout << "Training the Neural Network..." << std::endl;
    nn.train(X, y, 100, "mse");

    // Step 5: Test Predictions
    std::cout << "Testing predictions..." << std::endl;
    Matrix predictions = nn.predict(X);

    // Step 6: Print Results
    std::cout << "Predictions:\n";
    predictions.print();

    // Step 7: Evaluate Accuracy
    int correct = 0;
    for (int i = 0; i < y.getCols(); ++i) {
        int actual = (y(0, i) > 0.5) ? 1 : 0; // Convert to binary labels
        int predicted = (predictions(0, i) > 0.5) ? 1 : 0;
        if (actual == predicted) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / y.getCols();
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;

    double r2 = Metrics::r2(y, predictions);
    std::cout << "R^2: " << r2 << std::endl;

    std::cout << "Neural Network test complete." << std::endl;
    return 0;
}