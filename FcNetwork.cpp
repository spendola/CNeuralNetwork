//
//  FcNetwork.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "FcNetwork.hpp"

FcNetwork::FcNetwork(int input, int output)
{
    srand((unsigned)time(0));
    hiddenLayer = NULL;
    label = NULL;
    input = NULL;
    nIn = input;
    nOut = output;
    
    dataLoader = new DataLoader();
    remoteApi = new RemoteApi();
}

FcNetwork::~FcNetwork()
{
    SafeDelete(hiddenLayer);
    SafeDelete(dataLoader);
    SafeDelete(remoteApi);
    SafeDeleteArray(input);
    SafeDeleteArray(label);
}

DataLoader* FcNetwork::GetDataLoader()
{
    return dataLoader;
}

void FcNetwork::ConnectLayer(HiddenLayer* layer)
{
    if (hiddenLayer == NULL)
        hiddenLayer = layer;
    else
        hiddenLayer->ConnectChild(layer);
        
}

void FcNetwork::Start(bool enablePublishStatus)
{
    publishNetworkStatus = enablePublishStatus;
    double* parameters = new double[6];
    input = new double[dataLoader->trainingSampleSize]();
    label = new double[dataLoader->trainingLabelSize]();
    while(true)
    {
        Print("\nChoose an Option\n");
        Print("1) Adaptive Network Training\n");
        Print("2) Evaluate Network\n");
        Print("3) Load Parameters\n");
        Print("4) Show History\n");
        Print("9) Exit\n");
        switch (helpers::SafeCin())
        {
            case 1:
                Print("Enter Training Parameters(Epochs, Batch Size, Learning Rate, Regularization Rate, Decay Rate, Early Stop):\n");
                if(helpers::ParseParameters(parameters, 6))
                    AdaptiveTraining(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3], parameters[4], int(parameters[5]));
                break;
            case 2:
                
                break;
            case 3:
                LoadParameters(helpers::SelectFile("../Saved/", ".txt"), hiddenLayer->CountParameters(), true);
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
                dataLoader->GetRandomTrainingSample(input, label);
                double* output = hiddenLayer->FeedForward(input, 0, 0);
                hiddenLayer->BackPropagate(input, label);
                cost += neuralmath::quadraticcost(output, input, nOut);
            }
            
            ((FcLayer*)hiddenLayer)->UpdateParameters(batchSize, learningRate, lambda, dataLoader->numberOfTrainingSamples);
            cost = double(cost/batchSize);
            
            // Evaluate
            validationRate = EvaluateNetwork(validationRate > 90.0);
            validations += validationRate;
            Print("Epoch " + helpers::ToString(e) + " completed, cost: " + helpers::ToString(cost) + ", validation: " + helpers::ToString(validationRate) + "\n");
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
        dataLoader->GetValidationSample(subSample ? -1 : i, input, label);
        //dataLoader->PrintSentence(inputs, 32);
        double* output = hiddenLayer->FeedForward(input, 0, 0);
        pass += helpers::ParseOutput(output, nOut) == (int)label[0] ? 1.0 : 0.0;
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
        SaveParameters("../Saved/FcParameters.txt");
    }
    
    Publish("Adaptive Training Finished");
    if(!publishNetworkStatus)
        plot.SimplePlot(&progress, 600, 1200);
}


void FcNetwork::SaveParameters(std::string path)
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


void FcNetwork::LoadParameters(std::string path, int size, bool testValidation)
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

