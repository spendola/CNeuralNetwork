//
//  FcNetwork.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "FcNetwork.hpp"
#include <iostream>
#include <ctime>

FcNetwork::FcNetwork()
{
    std::cout << "Initializing Fully Connected Network\n";
    srand((unsigned)time(0));
    hiddenLayer = NULL;
    outputs = NULL;
    inputs = NULL;
    dataLoader = NULL;
    
    int choice = 0;    
    std::cout << "\nChoose a Network Model\n";
    std::cout << "1) Mnist Fully Connected Network\n";
    std::cout << "2) Sentiment Analysis Fully Connected Network\n";
    std::cout << "3) Exit\n";
    std::cin >> choice;
    
    switch (choice)
    {
        case 1:
            nIn = 784;
            nOut = 10;
            
            inputs = new double[nIn];
            outputs = new double[nOut];
            
            CreateLayer(nIn, 100);
            CreateLayer(100, nOut);
            
            dataLoader = new DataLoader();
            dataLoader->LoadMnistTrainingData("Training Data/Mnist/MnistTrainingData.txt", 50000, 784, 10);
            dataLoader->LoadMnistValidationData("Training Data/Mnist/MnistValidationData.txt", 10000, 784, 1);
            
            Start();
            SaveParameters("Saved/MnistParameters.txt");
            break;
            
        case 2:
            nIn = 32;
            nOut = 2;
            
            inputs = new double[nIn];
            outputs = new double[nOut];
            
            CreateLayer(nIn, 80);
            CreateLayer(80, nOut);
            
            dataLoader = new DataLoader();
            dataLoader->CreateDictionary("Training Data/Sentiment/TrainingData.txt");
            dataLoader->LoadSentimentTrainingData("Training Data/Sentiment/TrainingData Reinforced.txt", 32, 2);
            dataLoader->LoadSentimentValidationData("Training Data/Sentiment/ValidationData.txt", 32, 1);
            
            Start();
            SaveParameters("Saved/SentimentAnalysisParameters.txt");
            break;
            
        case 3:
            break;
            
        default:
            break;
    }
    
}

bool FcNetwork::CreateLayer(int input, int output)
{
    if (hiddenLayer == NULL)
    {
        hiddenLayer = new FcLayer(input, output);
        return true;
    }
    return hiddenLayer->CreateLayer(input, output);
}


void FcNetwork::Start()
{
    int choice;
    double* parameters = new double[6];
    while(true)
    {
        std::cout << "\nChoose an Option\n";
        std::cout << "1) Adaptive Network Training\n";
        std::cout << "2) Evaluate Network\n";
        std::cout << "3) Load Last Parameters\n";
        std::cout << "9) Exit\n";
        std::cin >> choice;
        switch (choice)
        {
            case 1:
                std::cout << "Enter Training Parameters(Epochs, Batch Size, Learning Rate, Regularization Rate, Decay Rate, Early Stop):\n";
                if(helpers::ParseParameters(parameters, 6))
                    AdaptiveTraining(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3], parameters[4], int(parameters[5]));
                break;
            case 3:
                LoadParameters("Saved/LastParameters.txt", hiddenLayer->CountParameters(), true);
                break;
            default:
                break;
        }
        if(choice == 9)
            break;
    }
}

double FcNetwork::TrainNetwork(int epochs, int batchSize, double learningRate, double lambda)
{
    if(hiddenLayer != NULL)
    {
        double validations = 0.0;
        double validationRate = 0.0;
        
        for (int e=0; e<epochs; e++)
        {
            // Train Network
            double cost = 0.0;
            for (int i=0; i<batchSize; i++)
            {
                dataLoader->GetRandomTrainingSample(inputs, outputs);
                double* output = hiddenLayer->FeedForward(inputs);
                
                hiddenLayer->BackPropagate(inputs, outputs);
                cost += QuadraticCost(output, inputs, nOut);
            }
            
            hiddenLayer->UpdateParameters(batchSize, learningRate, lambda, dataLoader->numberOfTrainingSamples);
            cost = double(cost/batchSize);
            
            // Evaluate
            validationRate = EvaluateNetwork(false);
            progress.push_back(validationRate);
            validations += validationRate;
            std::cout << "Epoch " << e << " Completed: lr: " << learningRate << ", cost: " << std::setprecision(4) << cost << ", ";
            std::cout << "validation: " << validationRate << "%\n";
        }
        return validations/(double)epochs;
    }
    return 0.0;
}

double FcNetwork::EvaluateNetwork(bool subSample)
{
    int pass = 0;
    int validationSamples = subSample ? dataLoader->numberOfValidationSamples/4 : dataLoader->numberOfValidationSamples;
    
    for (int i=0; i<validationSamples; i++)
    {
        dataLoader->GetValidationSample(subSample ? -1 : i, inputs, outputs);
        dataLoader->GetRandomValidationSample(inputs, outputs);
        double* output = hiddenLayer->FeedForward(inputs);
        pass += helpers::ParseOutput(output, nOut) ? 1.0 : 0.0;
    }
    return helpers::Percentage(pass, validationSamples);
}


void FcNetwork::AdaptiveTraining(int epochs, int batchSize, double learningRate, double lambda, double decayRate, int earlyStop)
{
    double adaptiveLearningRate = learningRate;
    double originalValidationRate = 1.0;
    int overfittingCount = 0;
    
    progress.clear();
    while (true)
    {
        double currentValidationRate = TrainNetwork(epochs, batchSize, adaptiveLearningRate, lambda);
        double delta = ((currentValidationRate/originalValidationRate)-1.0) * 100;
        std::cout << "\nCycle completed, average validation rate is  " << currentValidationRate << ", delta is " << delta << "%\n\n";
        
        overfittingCount = delta < 0.0 ? overfittingCount+1 : 0;
        adaptiveLearningRate = overfittingCount > earlyStop ? adaptiveLearningRate * decayRate : adaptiveLearningRate;
        if(overfittingCount > (earlyStop*2))
            break;
        
        originalValidationRate = currentValidationRate;
        SaveParameters("Saved/LastParameters.txt");
    }
    std::cout << "Adaptive Training Finished\n";
    std::cout << "Learning Rate: " << learningRate << ", Regularization Rate: " << lambda << "\n";
}


void FcNetwork::SaveParameters(std::string path)
{
    std::deque<double> parameters;
    if(hiddenLayer != NULL)
    {
        hiddenLayer->SaveParameters(&parameters);
        std::cout << "Saving Parameters (" << parameters.size() << " parameters found)\n";
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


void FcNetwork::LoadParameters(std::string path, int size, bool testValidation)
{
    std::cout << "Loading Parameters\n";
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
        std::cout << "Validation: " << EvaluateNetwork(false) << "%\n";
    }
}


double FcNetwork::QuadraticCost(double* x, double* y, int size)
{
    double distance = 0.0;
    for (int i=0; i<size; i++)
    {
        distance += pow((x[i] - y[i]), 2.0);
    }
    return 0.5 * (pow(sqrt(distance), 2.0));
}


FcNetwork::~FcNetwork()
{
    std::cout << "Cleaning up Fully Connected Network\n";
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDeleteArray(inputs);
    SafeDeleteArray(outputs);
}