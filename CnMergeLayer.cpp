//
//  CnMergeLayer.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 10/3/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "CnMergeLayer.hpp"

CnMergeLayer::CnMergeLayer(int width, int height)
{
    childLayer = NULL;
    siblingLayer = NULL;
    neuronSize = width * height;
    usedCapacity = 0;
    data = new double[neuronSize]();
    delta = new double[neuronSize]();
}

CnMergeLayer::~CnMergeLayer()
{
    SafeDeleteArray(data);
}

void CnMergeLayer::CleanUp()
{
    
}

int CnMergeLayer::LayerSize()
{
    return childLayer->LayerSize();
}

double* CnMergeLayer::LayerWeights()
{
    return childLayer->LayerWeights();
}

double* CnMergeLayer::FeedForward(double* input, int width, int height)
{
    for(int i=0; i<(width*height); i++)
        data[usedCapacity++] = input[i];
    
    if(usedCapacity == neuronSize)
    {
        //std::cout << "\n";
        //helpers::PrintArray("MergeLayer", data, neuronSize);
        //std::cout << "\n";
        
        usedCapacity = 0;
        pendingBackPropagation = true;
        return childLayer->FeedForward(data, neuronSize, 1);
    }
    return NULL;
}

void CnMergeLayer::BackPropagate(double *input, double *label)
{
    if(pendingBackPropagation)
    {
        childLayer->BackPropagate(input, label);
        pendingBackPropagation = false;
        pendingUpdateParameters = true;
    }
    
    // for each neuron in layer
    for(int i=0; i<neuronSize; i++)
    {
        delta[i] = 0.0;
        for(int e=0; e<childLayer->LayerSize(); e++)
            delta[i] += childLayer->LayerWeights()[(e*childLayer->LayerSize())+i] * childLayer->LayerDelta()[e];
    }
}

void CnMergeLayer::UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate)
{
    if(pendingUpdateParameters)
    {
        if(childLayer != NULL)
            childLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
        if(siblingLayer != NULL)
            siblingLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
        pendingUpdateParameters = false;
    }
}
