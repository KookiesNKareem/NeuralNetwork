#include <iostream>

#include "ActivationFunction.h"
#include "Matrix.h"
#include "Layer.h"
#include "NeuralNetwork.h"

void PropagationTest() {
    // Step 1: Initialize a layer with 2 inputs and 3 outputs
    Layer layer(2, 3, ActivationFunction::relu, ActivationFunction::reluDerivative);
    // Step 2: Create random input matrix (2 inputs, 4 samples)
    Matrix inputs = Matrix::random(2, 4, -1.0, 1.0);

    // Step 3: Perform forward propagation
    Matrix outputs = layer.forward(inputs);
    std::cout << "Forward Propagation Output:\n";
    outputs.print();

    // Step 4: Simulate a gradient of loss w.r.t. outputs (random for testing)
    Matrix dOutputs = Matrix::random(3, 4, -0.5, 0.5); // Same shape as outputs

    // Step 5: Perform backpropagation
    auto [dW, db, dA_prev] = layer.backward(dOutputs, inputs);
    std::cout << "Weight Gradient (dW):\n";
    dW.print();
    std::cout << "Bias Gradient (db):\n";
    db.print();
    std::cout << "Previous Layer Gradient (dA_prev):\n";
    dA_prev.print();

    // Step 6: Update weights and biases
    double learningRate = 0.01;
    layer.update(dW, db, learningRate);

    // Step 7: Verify parameter updates
    std::cout << "Updated Weights and Biases:\n";
    layer.printParameters();
}
void TestLayer() {
    // Initialize a layer with 2 inputs and 3 outputs
    Layer layer(2, 3, ActivationFunction::relu, ActivationFunction::reluDerivative);

    // Create random inputs: 2 features, 4 samples
    Matrix inputs = Matrix::random(2, 4, -1.0, 1.0);
    // Perform forward propagation
    Matrix outputs = layer.forward(inputs);
    std::cout << "Forward Propagation Output:\n";
    outputs.print();

    // Simulate a gradient of the loss w.r.t. the outputs
    Matrix dOutputs = Matrix::random(3, 4, -0.5, 0.5);

    // Perform backward propagation
    auto [dW, db, dA_prev] = layer.backward(dOutputs, inputs);

    // Update weights and biases
    double learningRate = 0.01;
    layer.update(dW, db, learningRate);

    // Print updated parameters
    std::cout << "Updated Layer Parameters:\n";
    layer.printParameters();
}
int main() {
    // Create the neural network
    NeuralNetwork nn(0.1);

    // Add layers
    nn.addLayer(2, 5, ActivationFunction::relu, ActivationFunction::reluDerivative); // Input: 2, Hidden: 3
    nn.addLayer(5, 1, ActivationFunction::sigmoid, ActivationFunction::sigmoidDerivative); // Hidden: 3, Output: 1

    // Generate random inputs and labels to test
    Matrix X = Matrix::random(2, 100, -1.0, 1.0); // 2 inputs, 100 samples
    Matrix y = Matrix::random(1, 100, 0, 1);      // Binary labels

    // Train the network for 100 epochs
    nn.train(X, y, 100);

    // Predict
    Matrix predictions = nn.predict(X);
    std::cout << "Predictions:\n";
    predictions.print();

    return 0;
}