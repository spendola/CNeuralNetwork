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
    FcNetwork* fcnet = new FcNetwork(784, 10);
    fcnet->ConnectLayer(new FcLayer(784, 100));
    fcnet->ConnectLayer(new FcLayer(100, 10));
    fcnet->GetDataLoader()->LoadMnistTrainingData("../Training Data/Mnist/MnistTrainingData.txt", 50000, 784, 10);
    fcnet->GetDataLoader()->LoadMnistValidationData("../Training Data/Mnist/MnistValidationData.txt", 10000, 784, 1);
    fcnet->Start(publishNetworkStatus);
    SafeDelete(fcnet);
}

void NetworkManager::SentAnalysisFcNetwork()
{
    Publish("Starting Fully Connected Network for Sentiment Analysis");
    FcNetwork* fcnet = new FcNetwork(32, 2);
    fcnet->ConnectLayer(new FcLayer(32, 100));
    fcnet->ConnectLayer(new FcLayer(100, 2));
    fcnet->GetDataLoader()->LoadStopWords("../Training Data/Sentiment/stopwords_en.txt");
    fcnet->GetDataLoader()->CreateTokenizedDictionary("../Training Data/Sentiment/TrainingData 65k.txt", false, false);
    fcnet->GetDataLoader()->LoadSentimentTrainingData("../Training Data/Sentiment/TrainingData.txt", 32, 2, false);
    fcnet->GetDataLoader()->LoadSentimentValidationData("../Training Data/Sentiment/ValidationData.txt", 32, 1, false);
    fcnet->Start(publishNetworkStatus);
    SafeDelete(fcnet);
}

void NetworkManager::LangModelRcNetwork()
{
    Publish("Starting Recursive Network for Language Modeling");
    RcNetwork* rcnet = new RcNetwork();
    rcnet->GetDataLoader()->CreateTokenizedDictionary("../Training Data/Sentiment/TrainingData 65k.txt", true, true);
    rcnet->GetDataLoader()->LoadLanguageModelTrainingData("../Training Data/Sentiment/TrainingData 65k.txt", 32);
    rcnet->CreateHiddenLayer(100, rcnet->GetDataLoader()->dictionarySize);
    rcnet->Start(publishNetworkStatus);
    SafeDelete(rcnet);
}

void NetworkManager::ImageRecCnNetwork()
{
    Publish("Starting Convolutional Network\n");
    CnNetwork* cnnet = new CnNetwork();
    
    //helpers::CalculateOutputSize(28, 28, 5, 5, 2, 2);
    
    HiddenLayer* fclayer10 = new FcLayer(100, 10);
    HiddenLayer* fclayer100 = new FcLayer(144*8, 100);
    HiddenLayer* mergelayer = new CnMergeLayer(144, 8);
    
    HiddenLayer* feature1 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling1 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature2 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling2 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature3 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling3 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature4 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling4 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature5 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling5 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature6 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling6 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature7 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling7 = new CnPoolingLayer(2, 2, 24, 24);
    HiddenLayer* feature8 = new CnFeatureMap(5, 5, 28, 28);
    HiddenLayer* pooling8 = new CnPoolingLayer(2, 2, 24, 24);
    
    fclayer100->ConnectChild(fclayer10);
    mergelayer->ConnectChild(fclayer100);
    

    feature1->SetLayerIndex(0);
    pooling1->SetLayerIndex(0);
    pooling1->ConnectChild(mergelayer);
    feature1->ConnectChild(pooling1);
    cnnet->ConnectSibling(feature1);
    

    feature2->SetLayerIndex(1);
    pooling2->SetLayerIndex(1);
    pooling2->ConnectChild(mergelayer);
    feature2->ConnectChild(pooling2);
    cnnet->ConnectSibling(feature2);
    

    feature3->SetLayerIndex(2);
    pooling3->SetLayerIndex(2);
    pooling3->ConnectChild(mergelayer);
    feature3->ConnectChild(pooling3);
    cnnet->ConnectSibling(feature3);
    
    feature4->SetLayerIndex(3);
    pooling4->SetLayerIndex(3);
    pooling4->ConnectChild(mergelayer);
    feature4->ConnectChild(pooling4);
    cnnet->ConnectSibling(feature4);
    
    feature5->SetLayerIndex(4);
    pooling5->SetLayerIndex(4);
    pooling5->ConnectChild(mergelayer);
    feature5->ConnectChild(pooling5);
    cnnet->ConnectSibling(feature5);
    
    feature6->SetLayerIndex(5);
    pooling6->SetLayerIndex(5);
    pooling6->ConnectChild(mergelayer);
    feature6->ConnectChild(pooling6);
    cnnet->ConnectSibling(feature6);
    
    feature7->SetLayerIndex(6);
    pooling7->SetLayerIndex(6);
    pooling7->ConnectChild(mergelayer);
    feature7->ConnectChild(pooling7);
    cnnet->ConnectSibling(feature7);
    
    feature8->SetLayerIndex(7);
    pooling8->SetLayerIndex(7);
    pooling8->ConnectChild(mergelayer);
    feature8->ConnectChild(pooling8);
    cnnet->ConnectSibling(feature8);
    
    
    cnnet->GetDataLoader()->LoadMnistTrainingData("../Training Data/Mnist/MnistTrainingData.txt", 50000, 784, 10);
    cnnet->GetDataLoader()->LoadMnistValidationData("../Training Data/Mnist/MnistValidationData.txt", 10000, 784, 1);
    
    cnnet->Start(publishNetworkStatus);
    SafeDelete(fclayer10);
    SafeDelete(fclayer100);
    SafeDelete(feature1);
    SafeDelete(feature2);
    SafeDelete(feature3);
    SafeDelete(feature4);
    SafeDelete(feature5);
    SafeDelete(feature6);
    SafeDelete(feature7);
    SafeDelete(feature8)
    SafeDelete(pooling1);
    SafeDelete(pooling2);
    SafeDelete(pooling3);
    SafeDelete(pooling4);
    SafeDelete(pooling5);
    SafeDelete(pooling6);
    SafeDelete(pooling7);
    SafeDelete(pooling8);
    SafeDelete(cnnet);
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
    bool finished = false;
    while(!finished)
    {
        Print("\nMain Menu\n");
        Print("1) Mnist Fully Connected Network\n");
        Print("2) Sentiment Analysis Fully Connected Network\n");
        Print("3) Language Modeling Recursive Network\n");
        Print("4) Convolutional Neural Network\n");
        Print("5) Options\n");
        Print("9) Exit\n");
        
        switch(helpers::SafeCin())
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
                ImageRecCnNetwork();
                break;
            case 5:
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
