//
//  FcLayer.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef FcLayer_hpp
#define FcLayer_hpp

#include <stdio.h>
#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include <sstream>
#include "deque"
#include "HiddenLayer.hpp"
#include "NeuralMath.hpp"
#include "Helpers.hpp"

class FcLayer : public HiddenLayer
{
private:
    double* delta_nabla_b;
    double* delta_nabla_w;
    double* nabla_b;
    double* nabla_w;
    
public:
    
    FcLayer(int input, int neurons);
    ~FcLayer();
    void InitializeWeightsAndBiases();
    
    double* FeedForward(double* input, int width, int height);
    void BackPropagate(double* input, double* label);
    void UpdateParameters(int batchSize, double learningRate, double lambda, int trainingSamples);
    
};

#endif /* FcLayer_hpp */
