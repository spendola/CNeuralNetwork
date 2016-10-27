//
//  CnMergeLayer.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 10/3/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef CnMergeLayer_hpp
#define CnMergeLayer_hpp

#include <stdio.h>
#include "HiddenLayer.hpp"
#include "Helpers.hpp"

class CnMergeLayer : public HiddenLayer
{
    
private:
    int usedCapacity;
    bool pendingBackPropagation;
    bool pendingUpdateParameters;
    double* data;
    
public:
    int LayerSize();
    double* LayerWeights();
    
    CnMergeLayer(int width, int height);
    ~CnMergeLayer();
    void CleanUp();
    
    double* FeedForward(double* input, int width, int height);
    void BackPropagate(double* input, double* label);
    void UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate);
    
};

#endif /* CnMergeLayer_hpp */
