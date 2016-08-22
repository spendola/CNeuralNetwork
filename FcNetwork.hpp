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
#include "deque"
#include "FcLayer.hpp"
#include "DataLoader.hpp"
#include "StopWatch.hpp"
#include "Helpers.hpp"
#include "History.hpp"

class FcNetwork
{
    
private:
    int nIn;
    int nOut;
    double* inputs;
    double* outputs;
    FcLayer* hiddenLayer;
    DataLoader* dataLoader;
    double QuadraticCost(double* x, double* y, int size);
    int ParseOutput(double* output);
    
    void SaveProgress();
    void SaveParameters(std::string path);
    void LoadParameters(std::string path, int size, bool testValidation);
    std::deque<double> progress;
    
public:
    
    FcNetwork();
    ~FcNetwork();
    
    void Start();
    bool CreateLayer(int input, int output);
    
    void AdaptiveTraining(int epochs, int batchSize, double learningRate, double lambda, double decayRate, int earlyStop);
    double TrainNetwork(int epochs, int batchSize, double learningRate, double lambda);
    double EvaluateNetwork(bool subSample);    
    
};



#endif /* FcNetwork_hpp */
