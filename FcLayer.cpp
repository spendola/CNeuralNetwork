//
//  FcLayer.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "FcLayer.hpp"


FcLayer::FcLayer(int input, int output)
{
    std::cout << "initializing fully connected layer with size " << input << ", " << output << "\n";
    nextLayer = NULL;

    nIn = input;
    nOut = output;
    
    if (nIn > 0 && nOut > 0)
    {
        activations = new double[nOut];
        zethas = new double[nOut];
        delta = new double[nOut];
        
        biases = new double[nOut];
        nabla_b = new double[nOut];
        delta_nabla_b = new double[nOut];
        
        weights = new double[nIn*nOut];
        nabla_w = new double[nIn*nOut];
        delta_nabla_w = new double[nIn*nOut];
        
        InitializeWeightsAndBiases();
    }
}

void FcLayer::InitializeWeightsAndBiases()
{
    std::default_random_engine de((unsigned)time(0));
    std::normal_distribution<double> sqrtnd(0.0, 1.0/(std::sqrt(nIn)));
    std::normal_distribution<double> nd(0.0, 1.0);
    for(int i = 0; i < (nIn*nOut); ++i)
    {
        weights[i] = sqrtnd(de);
        nabla_w[i] = 0.0;
        delta_nabla_w[i] = 0.0;
    }

    for (int i=0; i<nOut; i++)
    {
        biases[i] = nd(de);
        nabla_b[i] = 0.0;
        delta_nabla_b[i] = 0.0;
    }
}

bool FcLayer::CreateLayer(int input, int output)
{
    if (nextLayer == NULL)
    {
        nextLayer = new FcLayer(input, output);
        return true;
    }
    return nextLayer->CreateLayer(input, output);
}


double* FcLayer::FeedForward(double* in)
{
    for(int n=0; n<nOut; n++)
    {
        double wa = 0.0;
        for(int w=0; w<nIn; w++)
            wa += in[w] * weights[w+(n*nIn)];
        
        zethas[n] = wa + biases[n];
        activations[n] = neuralmath::sigmoid(zethas[n]);
    }
    
    if(nextLayer != NULL)
        return nextLayer->FeedForward(activations);
    return activations;
}

void FcLayer::BackPropagate(double* in, double* out)
{
    if (nextLayer == NULL)
    {
        for(int i=0; i<nOut; i++)
        {
            // Calculate error per neuron
            delta[i] = (activations[i] - out[i]);
            
            // Calculate error rate for biases
            nabla_b[i] = delta[i];
            delta_nabla_b[i] += nabla_b[i];

            // Calculate error rate for weights
            for(int e=0; e<nIn; e++)
            {
                nabla_w[(i*nIn)+e] = delta[i] * in[e];
                delta_nabla_w[(i*nIn)+e] += nabla_w[(i*nIn)+e];
            }
         }
    }
    else
    {
        nextLayer->BackPropagate(activations, out);
        
        // for each neuron in layer
        for (int i=0; i<nOut; i++)
        {
            delta[i] = 0.0;
            for (int e=0; e<nextLayer->nOut; e++)
                delta[i] += nextLayer->weights[i+(e*nextLayer->nOut)] * nextLayer->delta[e];
            delta[i] = delta[i] * neuralmath::sigmoidprime(zethas[i]);
            
            nabla_b[i] = delta[i];
            delta_nabla_b[i] += nabla_b[i];
            
            for (int e=0; e<nIn; e++)
            {
                nabla_w[(i*nIn)+e] = in[e] * delta[i];
                delta_nabla_w[(i*nIn)+e] += nabla_w[(i*nIn)+e];
            }
        }
    }
}

void FcLayer::UpdateParameters(int batchSize, double learningRate, double lambda, int trainingSamples)
{
    if(nextLayer != NULL)
        nextLayer->UpdateParameters(batchSize, learningRate, lambda, trainingSamples);
    
    for (int i=0; i<nOut; i++)
    {
        biases[i] = biases[i] - ((learningRate/double(batchSize))* delta_nabla_b[i]);
        delta_nabla_b[i] = 0.0;
    }
    
    for (int i=0; i<(nIn*nOut); i++)
    {
        weights[i] = ((1.0-((learningRate*lambda)/double(trainingSamples)))*weights[i])
                      -  ((learningRate/double(batchSize))*delta_nabla_w[i]);
        delta_nabla_w[i] = 0.0;
    }
}

int FcLayer::CountParameters()
{
    if(nextLayer != NULL)
        return (nOut + (nIn*nOut)) + nextLayer->CountParameters();
    return nOut + (nIn*nOut);
}

void FcLayer::SaveParameters(std::deque<double>* parameters)
{
    for (int w=0; w<(nIn*nOut); w++)
        parameters->push_back(weights[w]);
    for (int b=0; b<(nOut); b++)
        parameters->push_back(biases[b]);
    
    if(nextLayer != NULL)
        nextLayer->SaveParameters(parameters);
}

void FcLayer::LoadParameters(double* parameters, int start)
{
    int count = start;
    for (int w=0; w<(nIn*nOut); w++)
        weights[w] = parameters[count++];
    for (int b=0; b<(nOut); b++)
        biases[b] = parameters[count++];
    
    if(nextLayer != NULL)
        nextLayer->LoadParameters(parameters, count);
}

FcLayer::~FcLayer()
{
    std::cout << "Cleaning up Fully Connected Layer\n";
    SafeDelete(nextLayer);
    SafeDeleteArray(weights);
    SafeDeleteArray(biases);
    SafeDeleteArray(zethas);
    SafeDeleteArray(activations);
    SafeDeleteArray(delta);
    SafeDeleteArray(nabla_b);
    SafeDeleteArray(nabla_w);
}
