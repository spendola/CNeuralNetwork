//
//  FcLayer.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "FcLayer.hpp"


FcLayer::FcLayer(int input, int neurons)
{
    std::cout << "Initializing Fully Connected Layer with size " << input << ", " << neurons << "\n";
    
    childLayer = NULL;
    siblingLayer = NULL;
    inputSize = input;
    neuronSize = neurons;
    
    activations = new double[neuronSize];
    zethas = new double[neuronSize];
    delta = new double[neuronSize];
    
    biases = new double[neuronSize];
    nabla_b = new double[neuronSize];
    delta_nabla_b = new double[neuronSize];
    
    weights = new double[inputSize*neuronSize];
    nabla_w = new double[inputSize*neuronSize];
    delta_nabla_w = new double[inputSize*neuronSize];
    
    InitializeWeightsAndBiases();
}

FcLayer::~FcLayer()
{
    std::cout << "Deleting Fully Connected Layer\n";
    SafeDelete(childLayer);
    SafeDelete(siblingLayer);
    SafeDeleteArray(weights);
    SafeDeleteArray(biases);
    SafeDeleteArray(zethas);
    SafeDeleteArray(activations);
    SafeDeleteArray(delta);
    SafeDeleteArray(nabla_b);
    SafeDeleteArray(nabla_w);
}

void FcLayer::InitializeWeightsAndBiases()
{
    std::default_random_engine de((unsigned)time(0));
    std::normal_distribution<double> sqrtnd(0.0, 1.0/(std::sqrt(inputSize)));
    std::normal_distribution<double> nd(0.0, 1.0);
    for(int i = 0; i < (inputSize*neuronSize); ++i)
    {
        weights[i] = sqrtnd(de);
        nabla_w[i] = 0.0;
        delta_nabla_w[i] = 0.0;
    }

    for (int i=0; i<neuronSize; i++)
    {
        biases[i] = nd(de);
        nabla_b[i] = 0.0;
        delta_nabla_b[i] = 0.0;
    }
}

double* FcLayer::FeedForward(double* input, int width, int height)
{
    for(int n=0; n<neuronSize; n++)
    {
        double wa = 0.0;
        for(int w=0; w<inputSize; w++)
            wa += input[w] * weights[w+(n*inputSize)];
        
        zethas[n] = wa + biases[n];
        activations[n] = neuralmath::sigmoid(zethas[n]);
    }
    
    if(childLayer != NULL)
        return childLayer->FeedForward(activations, neuronSize, 1);
    
    return activations;
}

void FcLayer::BackPropagate(double* input, double* label)
{
    if (childLayer == NULL)
    {
        for(int i=0; i<neuronSize; i++)
        {
            // Calculate error per neuron
            delta[i] = (activations[i] - label[i]);
            
            // Calculate error rate for biases
            nabla_b[i] = delta[i];
            delta_nabla_b[i] += nabla_b[i];

            // Calculate error rate for weights
            for(int e=0; e<inputSize; e++)
            {
                nabla_w[(i*inputSize)+e] = delta[i] * input[e];
                delta_nabla_w[(i*inputSize)+e] += nabla_w[(i*inputSize)+e];
            }
         }
    }
    else
    {
        ((FcLayer*)childLayer)->BackPropagate(activations, label);
        
        // for each neuron in layer
        for (int i=0; i<neuronSize; i++)
        {
            delta[i] = 0.0;
            for (int e=0; e<childLayer->LayerSize(); e++)
            {
                delta[i] += childLayer->LayerWeights()[i+(e*childLayer->LayerSize())] * childLayer->LayerDelta()[e];
            }
            delta[i] = delta[i] * neuralmath::sigmoidprime(zethas[i]);
          
            nabla_b[i] = delta[i];
            delta_nabla_b[i] += nabla_b[i];
            
            for (int e=0; e<inputSize; e++)
            {
                nabla_w[(i*inputSize)+e] = input[e] * delta[i];
                delta_nabla_w[(i*inputSize)+e] += nabla_w[(i*inputSize)+e];
            }
        }
       
    }
}

void FcLayer::UpdateParameters(int batchSize, double learningRate, double lambda, int trainingSamples)
{
    if(childLayer != NULL)
        ((FcLayer*)childLayer)->UpdateParameters(batchSize, learningRate, lambda, trainingSamples);
    
    for (int i=0; i<neuronSize; i++)
    {
        biases[i] = biases[i] - ((learningRate/double(batchSize))* delta_nabla_b[i]);
        delta_nabla_b[i] = 0.0;
    }
    
    for (int i=0; i<(inputSize*neuronSize); i++)
    {
        weights[i] = ((1.0-((learningRate*lambda)/double(trainingSamples)))*weights[i])
                      -  ((learningRate/double(batchSize))*delta_nabla_w[i]);
        delta_nabla_w[i] = 0.0;
    }
}


