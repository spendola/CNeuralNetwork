//
//  FcLayer.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef FcLayer_hpp
#define FcLayer_hpp

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#include <stdio.h>
#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include <sstream>
#include "deque"

class FcLayer
{
private:
    double* delta_nabla_b;
    double* delta_nabla_w;
    double* nabla_b;
    double* nabla_w;
    double* biases;
    double* zethas;
    double* activations;
    
    
    FcLayer* nextLayer;
    double QuadraticCost(double* x, double* y);
    void QuadraticError(double* y, double* a, double* z);
    double Sigmoid(double z);
    double SigmoidPrime(double z);
    
public:
    int nIn;
    int nOut;
    
    double* delta;
    double* weights;
    
    FcLayer(int in, int out);
    ~FcLayer();
    bool CreateLayer(int input, int output);
    void InitializeWeightsAndBiases();
    
    double* FeedForward(double* in);
    void BackPropagate(double* in, double* out);
    void UpdateParameters(int batchSize, double learningRate, double lambda, int trainingSamples);
    
    int CountParameters();
    void SaveParameters(std::deque<double>* parameters);
    void LoadParameters(double* parameters, int start);
};

#endif /* FcLayer_hpp */
