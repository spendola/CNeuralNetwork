//
//  CnNetwork.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/21/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef CnNetwork_hpp
#define CnNetwork_hpp

#include <stdio.h>
#include "Helpers.hpp"
#include "DataLoader.hpp"
#include "RemoteApi.hpp"
#include "FcLayer.hpp"
#include "CnFeatureMap.hpp"
#include "CnPoolingLayer.hpp"
#include "CnMergeLayer.hpp"


class CnNetwork
{
    
private:
    HiddenLayer* hiddenLayer;
    DataLoader* dataLoader;
    RemoteApi* remoteApi;
    double* input;
    double* label;
    bool publishNetworkStatus;
    
    void SaveParameters(std::string path);
    void LoadParameters(std::string path, int size, bool testValidation);
    void Print(std::string str);
    void Publish(std::string str);
    
public:
    CnNetwork();
    ~CnNetwork();
    DataLoader* GetDataLoader();
    void ConnectSibling(HiddenLayer* layer);
    void ConnectChild(HiddenLayer* layer);
    
    void Start(bool enablePublishStatus);
    void TrainNetwork(int epochs, int batchSize, double learningRate, double regularizationRate);
    void AdaptiveTraining(int epochs, int batchSize, double learningRate, double regularizationRate);
    double EvaluateNetwork(bool subSample);
    
};

#endif /* CnNetwork_hpp */
