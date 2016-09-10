//
//  NetworkManager.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/7/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "NetworkManager.hpp"

NetworkManager::NetworkManager()
{
    publishNetworkStatus = false;
    remoteApi = new RemoteApi();
    Print("Neural Network Manager\n");
}

NetworkManager::~NetworkManager()
{
    SafeDelete(remoteApi);
}

void NetworkManager::Start()
{
    MainMenu();
}


void NetworkManager::MnistFcNetwork()
{
    Publish("Starting Fully Connected Network for Mnist");
    FcNetwork* fcnet = new FcNetwork();
    fcnet->CreateLayer(784, 100);
    fcnet->CreateLayer(100, 10);
    fcnet->GetDataLoader()->LoadMnistTrainingData("Training Data/Mnist/MnistTrainingData.txt", 50000, 784, 10);
    fcnet->GetDataLoader()->LoadMnistValidationData("Training Data/Mnist/MnistValidationData.txt", 10000, 784, 1);
    fcnet->Start(publishNetworkStatus);
    SafeDelete(fcnet);
}

void NetworkManager::SentAnalysisFcNetwork()
{
    Publish("Starting Fully Connected Network for Sentiment Analysis");
    FcNetwork* fcnet = new FcNetwork();
    fcnet->CreateLayer(32, 128);
    fcnet->CreateLayer(128, 2);
    fcnet->GetDataLoader()->CreateTokenizedDictionary("Training Data/Sentiment/TrainingData 65k.txt", 1);
    fcnet->GetDataLoader()->LoadSentimentTrainingData("Training Data/Sentiment/TrainingData 65k.txt", 32, 2);
    fcnet->GetDataLoader()->LoadSentimentValidationData("Training Data/Sentiment/ValidationData.txt", 32, 1);
    fcnet->Start(publishNetworkStatus);
    SafeDelete(fcnet);
}

void NetworkManager::LangModelRcNetwork()
{
    Publish("Starting Recursive Network for Language Modeling");
    RcNetwork* rcnet = new RcNetwork();
    rcnet->GetDataLoader()->CreateTokenizedDictionary("Training Data/Sentiment/TrainingData Reinforced.txt", 1);
    rcnet->GetDataLoader()->LoadLanguageModelTrainingData("Training Data/Sentiment/TrainingData Reinforced.txt", 32);
    rcnet->CreateHiddenLayer(100, rcnet->GetDataLoader()->dictionarySize);
    rcnet->Start(publishNetworkStatus);
    SafeDelete(rcnet);
}

void NetworkManager::ListenForRemote()
{
    while(true)
    {
        std::string message = remoteApi->FetchMessage();
        std::vector<double> instruction = helpers::ParseInstruction(message);
        
        switch ((int)instruction[0])
        {
            case 1:
                MnistFcNetwork();
                break;
            case 2:
                SentAnalysisFcNetwork();
                break;
            case 3:
                LangModelRcNetwork();
                break;
        }
        sleep(1);
    }
}

void NetworkManager::CleanTemporaryFiles()
{
    
}

void NetworkManager::MainMenu()
{
    int choice = 0;
    bool finished = false;
    while(!finished)
    {
        Print("\nMain Menu\n");
        Print("1) Mnist Fully Connected Network\n");
        Print("2) Sentiment Analysis Fully Connected Network\n");
        Print("3) Language Modeling Recursive Network\n");
        Print("4) Options\n");
        Print("9) Exit\n");
        std::cin >> choice;
        
        switch(choice)
        {
            case 1:
                MnistFcNetwork();
                break;
            case 2:
                SentAnalysisFcNetwork();
                break;
            case 3:
                LangModelRcNetwork();
                break;
            case 4:
                OptionsMenu();
                break;
            case 9:
                finished = true;
                break;
            default:
                break;
        }
    }

}

void NetworkManager::OptionsMenu()
{
    int choice = 0;
    Print("\nOptions Menu\n");
    Print("1) Publish Network Status\n");
    Print("2) Listen For Remote Commands\n");
    Print("3) Clean Temporary Files\n");
    Print("9) Exit\n");
    std::cin >> choice;
    
    switch(choice)
    {
        case 1:
            publishNetworkStatus = !publishNetworkStatus;
            Print(publishNetworkStatus ? "Publish Network Status Enabled\n" : "Publish Network Status Disabled\n");
            break;
        case 2:
            subscribeToRemote = !subscribeToRemote;
            Print(subscribeToRemote ? "Subscribe To Remote Api Enabled\n" : "Subscribe To Remote Api Disabled\n");
            break;
        case 3:
            CleanTemporaryFiles();
            break;
        default:
            break;
    }
}

void NetworkManager::Print(std::string str)
{
    std::cout << str;
}

void NetworkManager::Publish(std::string str)
{
    std::cout << str;
    if(publishNetworkStatus)
        remoteApi->PublishMessage(str);
}

std::vector<double> NetworkManager::Fetch()
{
    if(subscribeToRemote)
    {
        std::string instruction = remoteApi->FetchMessage();
        if(instruction.length() > 0)
            return helpers::ParseInstruction(instruction);
    }
    
    int input = 0;
    std::cin >> input;
    std::vector<double> output;
    output.push_back(input);
    return output;
}
