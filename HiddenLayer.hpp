//
//  HiddenLayer.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/23/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef HiddenLayer_hpp
#define HiddenLayer_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <deque>

class HiddenLayer
{
    
protected:
    HiddenLayer* childLayer;
    HiddenLayer* siblingLayer;

    int layerIndex;
    double* biases;
    double* weights;
    double* zethas;
    double* activations;
    double* delta;
    
    int inputSize;
    int inputWidth;
    int inputHeight;
    int neuronSize;
    int siblings;

public:    
    virtual int LayerSize();
    virtual double* LayerDelta();
    virtual double* LayerWeights();
    void SetLayerIndex(int index);
    
    virtual void CleanUp();
    virtual void InitializeWeightsAndBiases();
    virtual double* FeedForward(double* input, int width, int height);
    virtual void BackPropagate(double* input, double* label);
    virtual void UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate);
    
    void ConnectChild(HiddenLayer* layer);
    void ConnectSibling(HiddenLayer* layer);
    
    int CountParameters();
    void SaveParameters(std::deque<double>* parameters);
    void LoadParameters(double* parameters, int start);
};

#endif /* HiddenLayer_hpp */
