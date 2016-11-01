//
//  CnFeatureMap.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/20/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "CnFeatureMap.hpp"

CnFeatureMap::CnFeatureMap(int width, int height, int inputwidth, int inputheight)
{
    std::cout << "Initializing FeatureMap with size " << width << ", " << height << " ";
    childLayer = NULL;
    siblingLayer = NULL;
    
    featureSize = width*height;
    inputWidth = inputwidth;
    inputHeight = inputheight;
    inputSize = inputWidth*inputHeight;
    neuronSize = 1;
    featureWidth = width;
    featureHeight = height;
    
    horizontalSteps = inputWidth - featureWidth + 1;
    verticalSteps = inputHeight - featureHeight + 1;
    zethas = new double[horizontalSteps*verticalSteps]();
    activations = new double[horizontalSteps*verticalSteps]();
    delta = new double[horizontalSteps*verticalSteps]();
    
    weights = new double[featureSize*neuronSize]();
    biases = new double[neuronSize]();
    delta_nabla_w = new double[featureSize*neuronSize]();
    delta_nabla_b = new double[neuronSize]();
    
    std::cout << "(output: " << horizontalSteps*verticalSteps << ")\n";
    
    //srand((unsigned)time(0));
    InitializeWeights();
}

CnFeatureMap::~CnFeatureMap()
{
    std::cout << "Deleting Feature Map\n";
    SafeDelete(childLayer);
    SafeDelete(siblingLayer);
    SafeDeleteArray(zethas);
    SafeDeleteArray(activations);
    SafeDeleteArray(delta);
    SafeDeleteArray(weights);
    SafeDeleteArray(biases);
    SafeDeleteArray(delta_nabla_w);
    SafeDeleteArray(delta_nabla_b);
}


void CnFeatureMap::CleanUp()
{
    //SafeDeleteArray(activations);
}

void CnFeatureMap::InitializeWeights()
{
    std::default_random_engine de((unsigned)time(0));
    std::normal_distribution<double> nd(0.0, 1.0);
    
    for(int i = 0; i < featureSize; ++i)
        weights[i] = nd(de);
    for(int i = 0; i < neuronSize; i++)
        biases[i] = nd(de);    
}


double* CnFeatureMap::FeedForward(double *input, int width, int height)
{
    // Scan
    for(int r=0; r<verticalSteps; r++)
    {
        for(int c=0; c<horizontalSteps; c++)
        {
            zethas[(r*horizontalSteps)+c] = 0.0;
            for(int fh=0; fh<featureHeight; fh++)
            {
                for(int fw=0; fw<featureWidth; fw++)
                {
                    int inputIndex = (r*width) + c + (fh*width) + fw;
                    int localIndex = (fh*featureWidth) + fw;
                    zethas[(r*horizontalSteps)+c] += weights[localIndex] * input[inputIndex];
                    if(zethas[(r*horizontalSteps)+c] != zethas[(r*horizontalSteps)+c])
                    {
                        std::cout << "Index: " << (r*horizontalSteps) + c << "\n";
                        std::cout << "Weight " << weights[localIndex] << "\n";
                        std::cout << "input " << weights[inputIndex] << "\n";
                    }
                }
            }
            zethas[(r*horizontalSteps)+c] += biases[0];
            activations[(r*horizontalSteps)+c] = neuralmath::sigmoid(zethas[(r*horizontalSteps)+c]);
        }
    }
    if(zethas[0] != zethas[0])
        helpers::PrintArray("Zethas on FeedForward", zethas, horizontalSteps*verticalSteps);
    //helpers::PrintArrayEx("FeatureMap" + helpers::ToString(layerIndex), featureMap, horizontalSteps*verticalSteps, 3);
    
    double* retValue = NULL;
    if(siblingLayer != NULL)
        siblingLayer->FeedForward(input, width, height);
    if(childLayer != NULL)
        retValue = childLayer->FeedForward(activations, horizontalSteps, verticalSteps);
    
    return retValue;
}

void CnFeatureMap::BackPropagate(double *input, double *label)
{
    if(childLayer != NULL)
    {
        childLayer->BackPropagate(input, label);
        for(int i=0; i<(horizontalSteps*verticalSteps); i++)
            delta[i] = childLayer->LayerDelta()[i] * neuralmath::sigmoidprime(zethas[i]);
        
        
        for(int r=0; r<verticalSteps; r++)
            for(int c=0; c<horizontalSteps; c++)
            {
                int inputIndex = (r*horizontalSteps) + c;
                nabla_b += delta[inputIndex];
                
                for(int fh=0; fh<featureHeight; fh++)
                    for(int fw=0; fw<featureWidth; fw++)
                    {
                        int localIndex = (fh*featureWidth) + fw;
                        nabla_w += weights[localIndex] * delta[inputIndex];
                    }
            }
        
        delta_nabla_b[0] = nabla_b / (verticalSteps*horizontalSteps);
        for(int i=0; i<(featureSize*neuronSize); i++)
            delta_nabla_w[i] = nabla_w / (verticalSteps*horizontalSteps);;
    }
    else
        std::cout << "Couldn't find child layer to backpropagate!!!!\n";
    
    if(siblingLayer != NULL)
        siblingLayer->BackPropagate(input, label);
}

void CnFeatureMap::UpdateParameters(int batchSize, int numberOfTrainingSamples, double learningRate, double regularizationRate)
{
    for (int i=0; i<(featureSize*neuronSize); i++)
    {
        weights[i] = ((1.0-((learningRate*regularizationRate)/double(numberOfTrainingSamples)))*weights[i])
        -  ((learningRate/double(batchSize))*delta_nabla_w[i]);
        
        if(weights[i] != weights[i])
            helpers::PrintArray("Delta Nabla W", delta_nabla_w, featureSize);
        
        delta_nabla_w[i] = 0.0;
    }
    
    for (int i=0; i<neuronSize; i++)
    {
        biases[i] = biases[i] - ((learningRate/double(batchSize))* delta_nabla_b[i]);
        delta_nabla_b[i] = 0.0;
    }
    
    if(childLayer != NULL)
        childLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
    if(siblingLayer != NULL)
        siblingLayer->UpdateParameters(batchSize, numberOfTrainingSamples, learningRate, regularizationRate);
    
}
