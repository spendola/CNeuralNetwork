//
//  RcNetwork.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef RcNetwork_hpp
#define RcNetwork_hpp

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#include <stdio.h>
#include "DataLoader.hpp"
#include "RcLayer.hpp"
#include "RemoteApi.hpp"
#include "Common/Helpers.hpp"

class RcNetwork
{
private:
    int nVocabulary;
    double* input;
    double* output;
    bool publishNetworkStatus;
    
    RcLayer* hiddenLayer;
    DataLoader* dataLoader;
    RemoteApi* remoteApi;
    
    double* VectorizeSample(double* sample, int length);
    double CalculateLoss(double* input, double* output, int inputSize);
    double TrainNetwork(int epochs, int batchSize, double learningRate);
    
    void SaveParameters(std::string path);
    void LoadParameters(std::string path, int size, bool testValidation);
    
    void Print(std::string str);
    void Publish(std::string str);
    int autoSave;
    
public:
    RcNetwork();
    ~RcNetwork();
    DataLoader* GetDataLoader();
    
    void Start(bool enablePublishStatus);
    bool CreateHiddenLayer(int neurons, int vocabularySize);
    void PredictNextWord(std::string str, int tolerance);
    
};

#endif /* RcNetwork_hpp */
