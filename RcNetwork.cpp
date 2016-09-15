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
    hiddenLayer = NULL;
    input = NULL;
    output = NULL;
    dataLoader = new DataLoader();
    remoteApi = new RemoteApi();
    autoSave = 30;
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
    Print("Starting Recurrent Neural Network");
    publishNetworkStatus = enablePublishStatus || true;;
    if(publishNetworkStatus)
    {
        remoteApi->PublishCommand("flushgraph");
        remoteApi->PublishCommand("lossgraph");
    }
    
    double averageLoss = TrainNetwork(100000, 5, 1.0);
    Print("Training Completed: " + helpers::ToString(averageLoss) + " average Loss");
}

double RcNetwork::TrainNetwork(int epochs, int batchSize, double learningRate)
{
    Publish("Adaptive Training Started on Recurrent Network");
    if(publishNetworkStatus)
    {
        remoteApi->PublishCommand("flushgraph");
        remoteApi->PublishCommand("lossgraph");
    }
    
    int countBeforeSave = 0;
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
            double* expected = VectorizeSample(label, sentenceLength);
            double* output = hiddenLayer->FeedForward(input, sentenceLength);
            
            for(int w=0; w<sentenceLength; w++)
            {
                double diff = output[ (w*dataLoader->dictionarySize) + int(label[w])];
                partialLoss += std::log(diff) * -1.0;
            }
            Loss += partialLoss/double(sentenceLength);
            hiddenLayer->BackPropagate(input, expected, sentenceLength, learningRate);
            
            SafeDeleteArray(sample);
            SafeDeleteArray(label);
            SafeDeleteArray(input);
            SafeDeleteArray(expected);
        }
        Loss = double(Loss / batchSize);
        
        Print("Epoch completed: " + helpers::ToString(Loss) + " average loss");
        if(publishNetworkStatus)
            remoteApi->PublishValue(Loss);
        
        countBeforeSave++;
        if(countBeforeSave == autoSave)
        {
            SaveParameters("../Saved/RcParameters.txt");
            countBeforeSave = 0;
        }
    }
    return Loss;
}

int RcNetwork::PredictNextWord(double* input)
{
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

void RcNetwork::SaveParameters(std::string path)
{
    std::deque<double> parameters;
    if(hiddenLayer != NULL)
    {
        hiddenLayer->SaveParameters(&parameters);
        Print("Saving parameters (" + helpers::ToString(parameters.size()) + " parameters found)");
        if(parameters.size() > 0)
        {
            std::ofstream writer;
            writer.open (path, std::ios::binary | std::ios::out);
            if(writer.is_open())
            {
                while(parameters.size()>0)
                {
                    writer.write(reinterpret_cast<char*>(&parameters.front()), sizeof(double));
                    parameters.pop_front();
                }
            }
            writer.close();
        }
    }
}


void RcNetwork::LoadParameters(std::string path, int size, bool testValidation)
{
    Print("Loading parameters from " + path);
    
    std::ifstream file (path, std::ios::binary);
    if(file.is_open())
    {
        double* parameters = new double[size];
        int i=0;
        char buffer[sizeof(double)];
        while(file.read(buffer, sizeof(double)))
        {
            parameters[i] = *((double*)buffer);
            i++;
        }
        std::cout << i << " parameters found\n";
        if(hiddenLayer != NULL)
            hiddenLayer->LoadParameters(parameters, 0);
        //std::cout << "Validation: " << EvaluateNetwork(false) << "%\n";
        
        SafeDeleteArray(parameters);
    }
    else
    {
        Print("ERROR: Unable to open file");
    }
    
}

void RcNetwork::Print(std::string str)
{
    std::cout << str << "\n";
}

void RcNetwork::Publish(std::string str)
{
    std::cout << str << "\n";
    if(publishNetworkStatus)
        remoteApi->PublishMessage(str);
}