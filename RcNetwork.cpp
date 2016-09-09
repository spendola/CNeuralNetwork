//
//  RcNetwork.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "RcNetwork.hpp"

RcNetwork::RcNetwork()
{
    std::cout << "Initializing Recursive Neural Network\n";
    hiddenLayer = NULL;
    input = NULL;
    output = NULL;
    dataLoader = new DataLoader();
}

RcNetwork::~RcNetwork()
{
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDeleteArray(input);
    SafeDeleteArray(output);
}

bool RcNetwork::CreateHiddenLayer(int neurons, int vocabularySize)
{
    nVocabulary = vocabularySize;
    hiddenLayer = new RcLayer(neurons, vocabularySize);
    return true;
}

DataLoader* RcNetwork::GetDataLoader()
{
    return dataLoader;
}

void RcNetwork::Start(bool enablePublishStatus)
{
    std::cout << "Starting Recurrent Neural Network\n";
    publishNetworkStatus = enablePublishStatus;
    double loss = TrainNetwork(100, 30);
    std::cout << "Total Loss = " << loss << "\n";
}

double RcNetwork::TrainNetwork(int epochs, int batchSize)
{
    if(publishNetworkStatus)
    {
        remote::PublishMessage("Adaptive Training Started");
        remote::PublishCommand("lossgraph");
    }
    
    double Loss = 0.0;
    for(int epoch=0; epoch<epochs; epoch++)
    {
        Loss = 0.0;
        for (int batch=0; batch<batchSize; batch++)
        {
            double partialLoss = 0.0;
            double* sample = new double[32]();
            double* label = new double[32]();
            int sentenceLength = dataLoader->GetLanguageTrainingSample(sample, label);
        
            double* input = VectorizeSample(sample, sentenceLength);
            double* output = hiddenLayer->FeedForward(input, sentenceLength);
            
            for(int w=0; w<sentenceLength; w++)
            {
                double diff = output[ (w*dataLoader->dictionarySize) + int(label[w])];
                if(diff != diff)
                {
                    std::cout << "NAAAAANNNN\n";
                    helpers::PrintLabeledArray("output", output, nVocabulary*sentenceLength);
                }
                partialLoss += std::log(diff) * -1.0;
            }
            Loss += partialLoss/double(sentenceLength);
            hiddenLayer->BackPropagate(sentenceLength, 0.1);
        }
        Loss = double(Loss / batchSize);
        std::cout << "Epoch completed: " << Loss << " average loss\n";
        if(publishNetworkStatus)
            remote::PublishValue(Loss);
    }
    return Loss;
}

int RcNetwork::PredictNextWord(double* input)
{
    //double* output = hiddenLayer->FeedForward(input, 8);
    return 0;
}

double RcNetwork::CalculateLoss(double* input, double* expected, int inputSize)
{
    double loss = 0.0;
    double* output = hiddenLayer->FeedForward(input, 0);
    for(int i=0; i<inputSize; i++)
    {
        double diff = output[(i*nVocabulary) + int(expected[i])];
        loss += std::log(diff) * -1.0;
    }
    return loss;
}

double* RcNetwork::VectorizeSample(double* sample, int length)
{
    double* vectorized = new double[nVocabulary*length]();
    for(int i=0; i<length; i++)
        vectorized[(i*nVocabulary)+int(sample[i])] = 1.0;
    
    return vectorized;
}