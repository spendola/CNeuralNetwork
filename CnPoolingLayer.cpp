//
//  CnPoolingLayer.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/23/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "CnPoolingLayer.hpp"

CnPoolingLayer::CnPoolingLayer(int width, int height, int inputwidth, int inputheight)
{
    std::cout << "Initializing PoolingLayer with size " << width << ", " << height << " ";
    childLayer = NULL;
    siblingLayer = NULL;
    weights = NULL;
    biases = NULL;
    
    poolingWidth = width;
    poolingHeight = height;
    inputWidth = inputwidth;
    inputHeight = inputheight;
    horizontalSteps = inputWidth / poolingWidth;
    verticalSteps = inputWidth / poolingHeight;
    
    neuronSize = horizontalSteps*verticalSteps;
    inputSize = inputWidth * inputHeight;

    poolingMap = new double[neuronSize];
    delta = new double[inputSize];
    std::cout << "(output: " << neuronSize << ")\n";
}

CnPoolingLayer::~CnPoolingLayer()
{
    std::cout << "Deleting Pooling Layer\n";
    SafeDelete(childLayer);
    SafeDelete(siblingLayer);
}

void CnPoolingLayer::CleanUp()
{
    
}

double* CnPoolingLayer::LayerWeights()
{
    return childLayer->LayerWeights();
}


double* CnPoolingLayer::FeedForward(double* input, int width, int height)
{
    // Scan
    for(int r=0; r<verticalSteps; r++)
    {
        for(int c=0; c<horizontalSteps; c++)
        {
            double maxPoolingActivation = 0.0;
            for(int fh=0; fh<poolingHeight; fh++)
            {
                for(int fw=0; fw<poolingWidth; fw++)
                {
                    int inputIndex = (r*width*poolingHeight) + (c*poolingWidth) + (fh*width) + fw;
                    maxPoolingActivation = maxPoolingActivation < input[inputIndex] ? input[inputIndex] : maxPoolingActivation;
                }
            }
            poolingMap[(r*horizontalSteps)+c] = maxPoolingActivation;
        }
    }
    
    
    double* retValue = NULL;
    if(siblingLayer != NULL)
        siblingLayer->FeedForward(input, width, height);
    if(childLayer != NULL)
        retValue = childLayer->FeedForward(poolingMap, horizontalSteps, verticalSteps);
    
    return retValue;
}

void CnPoolingLayer::BackPropagate(double *input, double *label)
{
    //std::cout << "BackPropagating PoolingLayer " << layerIndex << "\n";
    if(childLayer != NULL)
    {
        childLayer->BackPropagate(input, label);
        for(int r=0; r<verticalSteps; r++)
            for(int c=0; c<horizontalSteps; c++)
            {
                int poolIndex = c+(r*horizontalSteps);
                for(int h=0; h<poolingHeight; h++)
                    for(int w=0; w<poolingWidth; w++)
                    {
                        int expandIndex = (c*poolingWidth) + (r*horizontalSteps*poolingWidth*poolingHeight) + (h*horizontalSteps*poolingWidth) + w;
                        delta[expandIndex] = childLayer->LayerDelta()[(layerIndex*neuronSize) + poolIndex];
                    }
            }
    }
    else
        std::cout << "Could not find child layer to Backpropagate\n";
    
    if(siblingLayer != NULL)
        siblingLayer->BackPropagate(input, label);
}

void CnPoolingLayer::UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate)
{
    if(childLayer != NULL)
        childLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
    if(siblingLayer != NULL)
        siblingLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
}
