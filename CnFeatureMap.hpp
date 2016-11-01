//
//  CnFeatureMap.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/20/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef CnFeatureMap_hpp
#define CnFeatureMap_hpp

#include <stdio.h>
#include <random>
#include <ctime>
#include <cmath>
#include "HiddenLayer.hpp"
#include "Helpers.hpp"
#include "NeuralMath.hpp"
#include "CnPoolingLayer.hpp"

class CnFeatureMap : public HiddenLayer
{
    
private:
    //double* featureMap;
    double nabla_b;
    double nabla_w;
    double* delta_nabla_w;
    double* delta_nabla_b;
    
    int featureSize;
    int featureWidth;
    int featureHeight;
    int horizontalSteps;
    int verticalSteps;
    void InitializeWeights();
    
public:
    CnFeatureMap(int width, int height, int inputwidth, int inputheight);
    ~CnFeatureMap();
    void CleanUp();
    
    double* FeedForward(double* input, int width, int height);
    void BackPropagate(double* input, double* label);
    void UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate);
    
};


#endif /* CnFeatureMap_hpp */
