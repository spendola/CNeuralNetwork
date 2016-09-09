//
//  FcNetwork.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef FcNetwork_hpp
#define FcNetwork_hpp

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#include <stdio.h>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <ctime>
#include "deque"
#include "FcLayer.hpp"
#include "DataLoader.hpp"
#include "Common/StopWatch.hpp"
#include "Common/Helpers.hpp"
#include "OpenCvPlot.hpp"

class FcNetwork
{
    
private:
    int nIn;
    int nOut;
    double* inputs;
    double* outputs;
    bool publishNetworkStatus;
    
    FcLayer* hiddenLayer;
    DataLoader* dataLoader;
    
    void SaveProgress();
    void SaveParameters(std::string path);
    void LoadParameters(std::string path, int size, bool testValidation);
    
public:
    
    FcNetwork();
    ~FcNetwork();
    
    void Start(bool enablePublishStatus);
    DataLoader* GetDataLoader();
    bool CreateLayer(int input, int output);
    
    void AdaptiveTraining(int epochs, int batchSize, double learningRate, double lambda, double decayRate, int earlyStop);
    double TrainNetwork(int epochs, int batchSize, double learningRate, double lambda);
    double EvaluateNetwork(bool subSample);    
    
};



#endif /* FcNetwork_hpp */
