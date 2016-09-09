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
    std::cout << "Neural Network Manager\n";
}

NetworkManager::~NetworkManager()
{
    
}

void NetworkManager::Start()
{
    MainMenu();
}


void NetworkManager::MnistFcNetwork()
{
    if(publishNetworkStatus)
        remote::PublishMessage("Starting Fully Connected Network for Mnist");
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
    if(publishNetworkStatus)
        remote::PublishMessage("Starting Fully Connected Network for Sentiment Analysis");
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
    if(publishNetworkStatus)
        remote::PublishMessage("Starting Recursive Network for Language Modeling");
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
        std::string dto = remote::FetchMessage("");
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
        std::cout << "\nMain Menu\n";
        std::cout << "1) Mnist Fully Connected Network\n";
        std::cout << "2) Sentiment Analysis Fully Connected Network\n";
        std::cout << "3) Language Modeling Recursive Network\n";
        std::cout << "4) Options\n";
        std::cout << "9) Exit\n";
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
    std::cout << "\nOptions Menu\n";
    std::cout << "1) Publish Network Status\n";
    std::cout << "2) Listen For Remote Commands\n";
    std::cout << "3) Clean Temporary Files\n";
    std::cout << "9) Exit\n";
    std::cin >> choice;
    
    switch(choice)
    {
        case 1:
            publishNetworkStatus = !publishNetworkStatus;
            if(publishNetworkStatus)
                remote::PublishCommand("flushmessages");
            std::cout << (publishNetworkStatus ? "Publish Network Status Enabled\n" : "Publish Network Status Disabled\n");
            
            break;
        case 2:
            ListenForRemote();
            break;
        case 3:
            CleanTemporaryFiles();
            break;
        default:
            break;
    }
}