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
    parameters = new double[3];
    autoSave = 30;
}

RcNetwork::~RcNetwork()
{
    SafeDelete(remoteApi);
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDeleteArray(input);
    SafeDeleteArray(output);
    SafeDeleteArray(parameters);
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
    publishNetworkStatus = enablePublishStatus;
    while(true)
    {
        Print("\nChoose an Option");
        Print("1) Language Modeling Training");
        Print("2) Generate Sentence");
        Print("3) Load Parameters");
        Print("4) Exit");
        switch (helpers::SafeCin())
        {
            case 1:
                Print("Enter Training Parameters (epochs, batch size, learning rate)");
                if(helpers::ParseParameters(parameters, 3))
                    TrainNetwork(int(parameters[0]), int(parameters[1]), parameters[2]);
                break;
            case 2:
                GenerateSentence();
                break;
            case 3:
                LoadParameters(helpers::SelectFile("../Saved/", ".txt"), hiddenLayer->CountParameters(), true);
                break;
            case 9:
                return;
                break;
            default:
                break;
        }
    }
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
            output = NULL;
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

void RcNetwork::PredictNextWord(std::string str, int tolerance)
{
    int unknownToken = dataLoader->GetValueFromDictionry("[unknown_token]");
    int endToken = dataLoader->GetValueFromDictionry("[end_token]");
    int token = dataLoader->GetValueFromDictionry(str);
    
    if(token != unknownToken)
    {
        double* sentence = new double[32];
        sentence[0] = token;
        for(int i=0; i<32; i++)
        {
            Print(dataLoader->GetWordFromDictionary(sentence[i]) + " ");
            double* output = hiddenLayer->FeedForward(VectorizeSample(sentence, i+1), i+1);
            int nextWord = helpers::RandomInRange(output, nVocabulary, 5);
            if(nextWord != endToken)
                sentence[i+1] = nextWord;
            else
                break;
        }
    }
}

std::string RcNetwork::GenerateSentence()
{
    return "";
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
        {
            if(i == hiddenLayer->CountParameters())
            {
                hiddenLayer->LoadParameters(parameters, 0);
                Print("Process completed");
            }
            else
                Print("ERROR: invalid number of parameters");
        }
        
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