cmake_minimum_required(VERSION 3.30)
project(NN)

set(CMAKE_CXX_STANDARD 20)

add_executable(NN main.cpp Matrix.cpp Layer.cpp ActivationFunction.cpp Loss.cpp Metrics.cpp NeuralNetwork.cpp)