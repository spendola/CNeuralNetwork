//
//  FcNetwork.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef FcNetwork_hpp
#define FcNetwork_hpp

#include <stdio.h>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <ctime>
#include "deque"
#include "FcLayer.hpp"
#include "DataLoader.hpp"
#include "RemoteApi.hpp"
#include "StopWatch.hpp"
#include "Helpers.hpp"
#include "OpenCvPlot.hpp"

class FcNetwork
{
    
private:
    int nIn;
    int nOut;
    double* input;
    double* label;
    bool publishNetworkStatus;
    
    HiddenLayer* hiddenLayer;
    DataLoader* dataLoader;
    RemoteApi* remoteApi;
    
    void SaveProgress();
    void SaveParameters(std::string path);
    void LoadParameters(std::string path, int size, bool testValidation);
    
    void Print(std::string str);
    void Publish(std::string str);
    
public:
    
    FcNetwork(int input, int output);
    ~FcNetwork();
    
    void Start(bool enablePublishStatus);
    DataLoader* GetDataLoader();
    void ConnectLayer(HiddenLayer* layer);
    
    void AdaptiveTraining(int epochs, int batchSize, double learningRate, double lambda, double decayRate, int earlyStop);
    double TrainNetwork(int epochs, int batchSize, double learningRate, double lambda);
    double EvaluateNetwork(bool subSample);    
    
};



#endif /* FcNetwork_hpp */
