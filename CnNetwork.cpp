//
//  CnNetwork.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/21/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "CnNetwork.hpp"

CnNetwork::CnNetwork()
{
    std::cout << "Initializing Convolutional Neural Network\n";
    hiddenLayer = NULL;
    input = NULL;
    label = NULL;
    dataLoader = new DataLoader();
    remoteApi = new RemoteApi();
}

CnNetwork::~CnNetwork()
{
    std::cout << "Deleting Convolutional Network\n";
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDelete(remoteApi);
    SafeDeleteArray(input);
    SafeDeleteArray(label);
}

DataLoader* CnNetwork::GetDataLoader()
{
    return dataLoader;
}

void CnNetwork::ConnectSibling(HiddenLayer* layer)
{
    if(hiddenLayer == NULL)
        hiddenLayer = layer;
    else
        hiddenLayer->ConnectSibling(layer);
}

void CnNetwork::ConnectChild(HiddenLayer* layer)
{
    if(hiddenLayer == NULL)
        hiddenLayer = layer;
    else
        hiddenLayer->ConnectChild(layer);
}

void CnNetwork::Start(bool enablePublishStatus)
{
    publishNetworkStatus = enablePublishStatus;
    double* parameters = new double[4];
    input = new double[dataLoader->trainingSampleSize]();
    label = new double[dataLoader->trainingLabelSize]();

    while(true)
    {
        Print("\nChoose an Option\n");
        Print("1) Train Network\n");
        Print("2) Adaptive Network Training\n");
        Print("3) Load Parameters\n");
        Print("9) Exit\n");
        switch (helpers::SafeCin())
        {
            case 1:
                Print("Enter Training Parameters(Epochs, Batch Size, Learning Rate, Regularization Rate):\n");
                if(helpers::ParseParameters(parameters, 4))
                    TrainNetwork(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3]);
                break;
            case 2:
                Print("Enter Training Parameters(Epochs, Batch Size, Learning Rate, Regularization Rate):\n");
                if(helpers::ParseParameters(parameters, 4))
                    AdaptiveTraining(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3]);
                break;
            case 3:
                LoadParameters(helpers::SelectFile("../Saved/", ".txt"), hiddenLayer->CountParameters(), true);
                break;
            default:
                return;
                break;
        }
    }
}

void CnNetwork::TrainNetwork(int epochs, int batchSize, double learningRate, double regularizationRate)
{
    double* output;
    for(int epoch=0; epoch<epochs; epoch++)
    {
        for(int batch=0; batch<batchSize; batch++)
        {
            dataLoader->GetRandomTrainingSample(input, label);
            output = hiddenLayer->FeedForward(input, 28, 28);
            hiddenLayer->BackPropagate(input, label);
        }
        hiddenLayer->UpdateParameters(batchSize, dataLoader->numberOfTrainingSamples, learningRate, regularizationRate);
        Print("Epoch " + helpers::ToString(epoch) + " completed\n");
    }
}

void CnNetwork::AdaptiveTraining(int epochs, int batchSize, double learningRate, double regularizationRate)
{
    double validationRate = 0.0;
    Publish("Adaptive Training Started");
    Publish("Learning Rate: " + std::to_string(learningRate) + ", Regularization: " + std::to_string(regularizationRate));
    if(publishNetworkStatus)
    {
        remoteApi->PublishCommand("validationgraph");
        remoteApi->PublishCommand("flushgraph");
    }
    
    while(true)
    {
        double* output;
        for(int epoch=0; epoch<epochs; epoch++)
        {
            for(int batch=0; batch<batchSize; batch++)
            {
                dataLoader->GetRandomTrainingSample(input, label);
                output = hiddenLayer->FeedForward(input, 28, 28);
                hiddenLayer->BackPropagate(input, label);
            }
            hiddenLayer->UpdateParameters(batchSize, dataLoader->numberOfTrainingSamples, learningRate, regularizationRate);
            Print("Epoch " + helpers::ToString(epoch) + " completed\n");
        }
        Print("\nCycle Completed. Evaluating Network Accuracy\n");
        validationRate = EvaluateNetwork(true);
        if(publishNetworkStatus)
            remoteApi->PublishValue(validationRate);
        Print("Accuracy on Training Data: " + helpers::ToString(validationRate) + "\n");
        //SaveParameters("../Saved/CnParameters.txt");
    }
}

double CnNetwork::EvaluateNetwork(bool subSample)
{
    int pass = 0;
    int validationSamples = subSample ? dataLoader->numberOfValidationSamples/10 : dataLoader->numberOfValidationSamples;
    
    for (int i=0; i<validationSamples; i++)
    {
        dataLoader->GetValidationSample(subSample ? -1 : i, input, label);
        double* output = hiddenLayer->FeedForward(input, 0, 0);
        int response = (int)helpers::ParseOutput(output, 10);
        pass += response == (int)label[0] ? 1.0 : 0.0;
        std::cout << response << ", ";
    }
    std::cout << "\n";
    return helpers::Percentage(pass, validationSamples);
}

void CnNetwork::SaveParameters(std::string path)
{
    std::deque<double> parameters;
    if(hiddenLayer != NULL)
    {
        Print("Saving Parameters\n");
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


void CnNetwork::LoadParameters(std::string path, int size, bool testValidation)
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
                Print("Validation: " + helpers::ToString(EvaluateNetwork(false)));
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



void CnNetwork::Print(std::string str)
{
    std::cout << str;
}

void CnNetwork::Publish(std::string str)
{
    std::cout << str << "\n";
    if(publishNetworkStatus)
        remoteApi->PublishMessage(str);
}
