//
//  CnPoolingLayer.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/23/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef CnPoolingLayer_hpp
#define CnPoolingLayer_hpp

#include <stdio.h>
#include <iostream>
#include "HiddenLayer.hpp"
#include "Helpers.hpp"
#include "NeuralMath.hpp"

class CnPoolingLayer : public HiddenLayer
{
private:
    int poolingWidth;
    int poolingHeight;
    int horizontalSteps;
    int verticalSteps;
    
public:
    CnPoolingLayer(int width, int height, int inputwidth, int inputheight);
    ~CnPoolingLayer();
    void CleanUp();
    
    double* LayerWeights();
    
    double* FeedForward(double* input, int width, int height);
    void BackPropagate(double* input, double* label);
    void UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate);
};

#endif /* CnPoolingLayer_hpp */
