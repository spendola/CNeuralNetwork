//
//  RcLayer.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "RcLayer.hpp"

RcLayer::RcLayer(int neurons, int vocabularySize)
{
    nNeurons = neurons;
    nVocabulary = vocabularySize;
    
    weights_in = new double[nNeurons*nVocabulary]();
    weights_out = new double[nNeurons*nVocabulary]();
    weights_time = new double[nNeurons*nNeurons]();
    
    stepOutput = NULL;
    stepActivation = NULL;
    
    InitializeWeights(-(1/sqrt(double(nVocabulary))), (1/sqrt(double(nVocabulary))));
}

RcLayer::~RcLayer()
{
    SafeDeleteArray(weights_in);
    SafeDeleteArray(weights_out);
    SafeDeleteArray(weights_time);
}

void RcLayer::CleanUp()
{
    SafeDeleteArray(stepOutput);
    SafeDeleteArray(stepActivation);
}

void RcLayer::InitializeWeights(double lower_bound, double upper_bound)
{
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;
    for(int i=0; i < (nVocabulary*nNeurons); ++i)
    {
        weights_in[i] = unif(re);
        weights_out[i] = unif(re);
    }
    for(int i=0; i<(nNeurons*nNeurons); i++)
        weights_time[i] = unif(re);
}

double* RcLayer::FeedForward(double* in, int wordsInSentence)
{
    stepActivation = new double[wordsInSentence*nNeurons]();
    stepOutput = new double[wordsInSentence*nVocabulary]();
    
    // Propagate into hidden layer
    for(int w=0; w<wordsInSentence; w++)
    {
        if(!neuralmath::LayerPropagation(&stepActivation[w*nNeurons], in, weights_in, nNeurons, nVocabulary))
            std::cout << "ERROR in LayerPropagation";
        
        if(w > 0)
            if(!neuralmath::LayerPropagation(&stepActivation[w*nNeurons], &stepActivation[(w-1)*nNeurons], weights_time, nNeurons, nNeurons))
                std::cout << "ERROR in LayerPropagation";
        
        for(int n=0; n<nNeurons; n++)
            stepActivation[(w*nNeurons)+n] = std::tanh(stepActivation[(w*nNeurons)+n]);
    }
    
    // Propagate out of hidden layer
    for(int w=0; w<wordsInSentence; w++)
    {
        neuralmath::LayerPropagation(&stepOutput[w*nVocabulary], &stepActivation[w*nNeurons], weights_out, nVocabulary, nNeurons);
        neuralmath::softmax(&stepOutput[w*nVocabulary], nVocabulary);
    }
    return stepOutput;
}

void RcLayer::BackPropagate(double* in, double* expected, int wordsInSentence, double learningRate)
{
    int bptt_limit = 10;
    double* delta_U = new double[nNeurons*nVocabulary]();
    double* delta_V = new double[nNeurons*nVocabulary]();
    double* delta_W = new double[nNeurons*nNeurons]();
    
    double* delta_time = new double[nNeurons]();
    double* delta_output = new double[wordsInSentence*nVocabulary]();
    
    for(int i=0; i<(wordsInSentence*nVocabulary); i++)
        delta_output[i] = stepOutput[i] - (expected[i] * 1.0);
    
    for(int w=(wordsInSentence-1); w>=0; w--)
    {
        // Initial delta calculation
        neuralmath::TensorProduct(delta_V, &delta_output[w*nVocabulary], &stepActivation[w*nNeurons], nVocabulary, nNeurons);

        neuralmath::LayerBackpropagation(delta_time, &delta_output[w*nVocabulary], delta_V, nVocabulary, nNeurons);
        for(int i=0; i<nNeurons; i++)
            delta_time[i] = delta_time[i] * (1.0 - (stepActivation[(w*nNeurons)+i]*stepActivation[(w*nNeurons)+i]));
        
        // Backpropagation through time
        int end = std::max(w-bptt_limit, 1);
        for(int btt=(w-1); btt > (end-1); btt--)
        {
            neuralmath::TensorProduct(delta_W, delta_time, &stepActivation[btt], nNeurons, nNeurons);
            
            for(int v=0; v<nVocabulary; v++)
                for(int n=0; n<nNeurons; n++)
                    delta_U[(v*nNeurons)+n] += in[(btt*nVocabulary)+v]*delta_time[n];
            
            neuralmath::LayerBackpropagation(delta_time, &delta_output[w*nVocabulary], delta_V, nVocabulary, nNeurons);
            for(int i=0; i<nNeurons; i++)
                delta_time[i] = delta_time[i] * (1.0 - (stepActivation[((w-1)*nNeurons)+i]*stepActivation[((w-1)*nNeurons)+i]));
        }
        
    }
    
    // Update values
    for(int i=0; i<(nNeurons*nVocabulary); i++)
    {
        weights_in[i] = weights_in[i] - (learningRate * delta_U[i]);
        weights_out[i] = weights_out[i] - (learningRate * delta_V[i]);
    }
    
    for(int i=0; i<(nNeurons*nNeurons); i++)
    {
        weights_time[i] = weights_time[i] - (learningRate * delta_W[i]);
    }
    
    SafeDeleteArray(delta_U);
    SafeDeleteArray(delta_W);
    SafeDeleteArray(delta_V);
    SafeDeleteArray(delta_output);
    SafeDeleteArray(delta_time);
}

int RcLayer::CountParameters()
{
    return ((nVocabulary*nNeurons)*2) + nNeurons ;
}

void RcLayer::SaveParameters(std::deque<double>* parameters)
{
    for (int w=0; w<(nVocabulary*nNeurons); w++)
        parameters->push_back(weights_in[w]);
    for (int w=0; w<(nVocabulary*nNeurons); w++)
        parameters->push_back(weights_out[w]);
    for (int w=0; w<(nNeurons); w++)
        parameters->push_back(weights_time[w]);
}

void RcLayer::LoadParameters(double* parameters, int start)
{
    int count = start;
    for (int w=0; w<(nVocabulary*nNeurons); w++)
        weights_in[w] = parameters[count++];
    for (int w=0; w<(nVocabulary*nNeurons); w++)
        weights_out[w] = parameters[count++];
    for (int w=0; w<(nNeurons); w++)
        weights_time[w] = parameters[count++];
}
