//
//  HiddenLayer.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/23/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "HiddenLayer.hpp"


void HiddenLayer::CleanUp()
{
    // Nothing to do here
}

void HiddenLayer::InitializeWeightsAndBiases()
{
    // Nothing to do here
}

void HiddenLayer::SetLayerIndex(int index)
{
    layerIndex = index;
}

int HiddenLayer::LayerSize()
{
    return neuronSize;
}

double* HiddenLayer::LayerWeights()
{
    return weights;
}

double* HiddenLayer::LayerDelta()
{
    return delta;
}

void HiddenLayer::ConnectChild(HiddenLayer* layer)
{
    if(childLayer == NULL)
        childLayer = layer;
    else
        childLayer->ConnectChild(layer);
}

void HiddenLayer::ConnectSibling(HiddenLayer *layer)
{
    if(siblingLayer == NULL)
        siblingLayer = layer;
    else
        siblingLayer->ConnectSibling(layer);
}

double* HiddenLayer::FeedForward(double *input, int width, int height)
{
    if(childLayer != NULL)
        return childLayer->FeedForward(input, width, height);
    return NULL;
}

void HiddenLayer::BackPropagate(double* input, double* label)
{
    if(childLayer != NULL)
        childLayer->BackPropagate(input, label);
}

void HiddenLayer::UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate)
{
    if(childLayer != NULL)
        childLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
}

int HiddenLayer::CountParameters()
{
    if(childLayer != NULL)
        return (neuronSize + (inputSize*neuronSize)) + childLayer->CountParameters();
    return neuronSize + (inputSize*neuronSize);
}

void HiddenLayer::SaveParameters(std::deque<double>* parameters)
{
    for (int w=0; w<(inputSize*neuronSize); w++)
        parameters->push_back(weights[w]);
    for (int b=0; b<(neuronSize); b++)
        parameters->push_back(biases[b]);
    
    if(siblingLayer != NULL)
        siblingLayer->SaveParameters(parameters);
    if(childLayer != NULL)
        childLayer->SaveParameters(parameters);
}

void HiddenLayer::LoadParameters(double* parameters, int start)
{
    int count = start;
    for (int w=0; w<(inputSize*neuronSize); w++)
        weights[w] = parameters[count++];
    for (int b=0; b<(neuronSize); b++)
        biases[b] = parameters[count++];
    
    if(siblingLayer != NULL)
        siblingLayer->LoadParameters(parameters, count);
    if(childLayer != NULL)
        childLayer->LoadParameters(parameters, count);
}
