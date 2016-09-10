//
//  FcNetwork.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "FcNetwork.hpp"

FcNetwork::FcNetwork()
{
    srand((unsigned)time(0));
    hiddenLayer = NULL;
    outputs = NULL;
    inputs = NULL;
    
    dataLoader = new DataLoader();
    remoteApi = new RemoteApi();
}

FcNetwork::~FcNetwork()
{
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDelete(remoteApi);
    SafeDeleteArray(inputs);
    SafeDeleteArray(outputs);
}

DataLoader* FcNetwork::GetDataLoader()
{
    return dataLoader;
}

bool FcNetwork::CreateLayer(int input, int output)
{
    if (hiddenLayer == NULL)
    {
        nIn = input;
        nOut = output;
        hiddenLayer = new FcLayer(input, output);
        return true;
    }
    nOut = output;
    return hiddenLayer->CreateLayer(input, output);
}

void FcNetwork::Start(bool enablePublishStatus)
{
    publishNetworkStatus = enablePublishStatus;
    double* parameters = new double[6];
    inputs = new double[dataLoader->trainingSampleSize]();
    outputs = new double[dataLoader->trainingLabelSize]();
    while(true)
    {
        int choice = 0;
        Print("\nChoose an Option\n");
        Print("1) Adaptive Network Training\n");
        Print("2) Evaluate Network\n");
        Print("3) Load Last Parameters\n");
        Print("4) Show History\n");
        Print("9) Exit\n");
        std::cin >> choice;
        switch (choice)
        {
            case 1:
                Print("Enter Training Parameters(Epochs, Batch Size, Learning Rate, Regularization Rate, Decay Rate, Early Stop):\n");
                if(helpers::ParseParameters(parameters, 6))
                    AdaptiveTraining(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3], parameters[4], int(parameters[5]));
                break;
            case 2:
                
                break;
            case 3:
                LoadParameters("Saved/LastParameters.txt", hiddenLayer->CountParameters(), true);
                break;
            case 4:
                history::get();
                break;
            default:
                break;
        }
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
                
                cost += neuralmath::quadraticcost(output, inputs, nOut);
            }
            
            hiddenLayer->UpdateParameters(batchSize, learningRate, lambda, dataLoader->numberOfTrainingSamples);
            cost = double(cost/batchSize);
            
            // Evaluate
            validationRate = EvaluateNetwork(validationRate > 90.0);
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
    int validationSamples = subSample ? dataLoader->numberOfValidationSamples/10 : dataLoader->numberOfValidationSamples;
    
    for (int i=0; i<validationSamples; i++)
    {
        dataLoader->GetValidationSample(subSample ? -1 : i, inputs, outputs);
        double* output = hiddenLayer->FeedForward(inputs);
        pass += helpers::ParseOutput(output, nOut) == (int)outputs[0] ? 1.0 : 0.0;
    }
    return helpers::Percentage(pass, validationSamples);
}


void FcNetwork::AdaptiveTraining(int epochs, int batchSize, double learningRate, double lambda, double decayRate, int loops)
{
    std::deque<double> progress;
    OpenCvPlot plot;
    int earlyStop = 10;
    double adaptiveLearningRate = learningRate;
    double originalValidationRate = 1.0;
    int overfittingCount = 0;
    
    Publish("Adaptive Training Started");
    Publish("Learning Rate: " + std::to_string(learningRate) + ", Regularization: " + std::to_string(lambda));

    if(publishNetworkStatus)
    {
        remoteApi->PublishCommand("validationgraph");
        remoteApi->PublishCommand("flushgraph");
    }

    progress.clear();
    for (int i=0; i<loops; i++)
    {
        double currentValidationRate = TrainNetwork(epochs, batchSize, adaptiveLearningRate, lambda);
        double delta = ((currentValidationRate/originalValidationRate)-1.0) * 100;
        Print("\nCycle completed, average validation rate is  " + helpers::ToString(currentValidationRate) +
              "%, delta is " + helpers::ToString(delta) + "%\n\n");
        
        if(publishNetworkStatus)
            remoteApi->PublishValue(currentValidationRate);
        else
            progress.push_back(currentValidationRate);
    
        overfittingCount = delta < 0.0 ? overfittingCount+1 : 0;
        if(overfittingCount > earlyStop)
        {
            adaptiveLearningRate = adaptiveLearningRate * decayRate;
            overfittingCount = 0;
        }
        
        originalValidationRate = currentValidationRate;
        SaveParameters("Saved/LastParameters.txt");
    }
    
    
    if(publishNetworkStatus)
        remoteApi->PublishMessage("Adaptive Training Finished");
    else
        plot.SimplePlot(&progress, 600, 1200);
    
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
    std::cout << "Loading Parameters: " << path << "\n";
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

void FcNetwork::Print(std::string str)
{
    std::cout << str;
}

void FcNetwork::Publish(std::string str)
{
    std::cout << str << "\n";
    if(publishNetworkStatus)
        remoteApi->PublishMessage(str);
}

