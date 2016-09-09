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
    SafeDelete(stepActivation);
    stepActivation = new double[(wordsInSentence+1)*nNeurons]();
    
    SafeDeleteArray(stepOutput);
    stepOutput = new double[wordsInSentence*nVocabulary]();
    
    // Propagare into hidden layer
    for(int w=0; w<wordsInSentence; w++)
    {
        for(int n=0; n<nNeurons; n++)
        {
            for(int e=0; e<nVocabulary; e++)
                stepActivation[((w+1)*nNeurons)+n] += (in[(w*nVocabulary)+e] * weights_in[(n*nVocabulary)+e]);
            
            double prevActivation = 0.0;
            for(int e=0; e<nNeurons; e++)
                prevActivation += weights_time[(n*nNeurons)+e] * stepActivation[((w)*nNeurons)+n];
            
            stepActivation[((w+1)*nNeurons)+n] = std::tanh(stepActivation[((w+1)*nNeurons)+n] + prevActivation);
        }
        if(helpers::CheckForNan(&stepActivation[(w+1)*nNeurons], nNeurons))
            std::cout << "NAN\n";
    }
    
    // Propagate out of hidden layer
    for(int w=0; w<wordsInSentence; w++)
    {
        for(int v=0; v<nVocabulary; v++)
            for(int a=0; a<nNeurons; a++)
                stepOutput[(w*nVocabulary)+v] += (stepActivation[((w+1)*nNeurons)+a] * weights_out[(v*nNeurons)+a]);
        
        neuralmath::softmax(&stepOutput[w*nVocabulary], nVocabulary);
        
        if(helpers::CheckForNan(&stepOutput[w*nVocabulary], nVocabulary))
        {
            helpers::PrintLabeledArray("StepOut After Softmax", &stepOutput[w*nVocabulary], nVocabulary);
            helpers::PrintLabeledArray("StepActivation", &stepActivation[(w+1)*nNeurons], nNeurons);
            helpers::PrintLabeledArray("StepActivation", weights_out, nVocabulary*nNeurons);
            std::cout << "NAN\n";
        }
    }
    return stepOutput;
}

void RcLayer::BackPropagate(int wordsInSentence, double learningRate)
{
    double* delta_weights_in = new double[nNeurons*nVocabulary]();
    double* delta_weights_out = new double[nNeurons*nVocabulary]();
    double* delta_weights_time = new double[nNeurons]();
    double* delta_time = new double[nVocabulary*wordsInSentence];
    double* delta_output = new double[wordsInSentence*nVocabulary]();

    
    for(int i=0; i<(wordsInSentence*nVocabulary); i++)
        delta_output[i] = stepOutput[i] - 1.0;
    
    //  0     1      2      3
    // 0-9, 10-19, 20-29, 30-39, 40-49
    
    
    for(int w=(wordsInSentence-1); w>-1; w--)
    {
        int activation = (w+1)*nNeurons;
        int output = w*nVocabulary;
        
        for(int v=0; v<nVocabulary; v++)
        {
            for(int n=0; n<nNeurons; n++)
            {
                delta_weights_out[(v*nNeurons)+n] = delta_output[output+n] * stepActivation[activation+n];
                double a = weights_out[(v*nNeurons)+n];
                double b = delta_output[output+n];
                double c = stepActivation[activation+n];
                delta_time[n] = (a*b) + (1.0 - (c*c));
            }
        }
        //helpers::PrintArray("delta_weights_out", delta_weights_out, nNeurons);
        //helpers::PrintLabeledArray("delta_time", delta_time, nNeurons);
        
        for(int n=0; n<nNeurons; n++)
        {
            for(int i=0; i<nVocabulary; i++)
            {
                delta_weights_time[n] += delta_time[i+(n*nNeurons)] * stepActivation[i];
                delta_weights_in[(n*nNeurons)+i] = delta_time[i];
            }
        }
    }
    
    
    
    // Update values
    for(int i=0; i<(nNeurons*nVocabulary); i++)
    {
        weights_in[i] = weights_in[i] - (learningRate * delta_weights_in[i]);
        weights_out[i] = weights_out[i] - (learningRate * delta_weights_out[i]);
    }
    
    for(int i=0; i<(nNeurons*nNeurons); i++)
    {
        weights_time[i] = weights_time[i] - (learningRate * delta_time[i]);
    }
}

int RcLayer::CountParameters()
{
    return nNeurons + (nVocabulary*nNeurons);
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
