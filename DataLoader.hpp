//
//  DataLoader.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef DataLoader_hpp
#define DataLoader_hpp

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

class DataLoader
{
private:
    int trainingLabelSize;
    int trainingSampleSize;
    double* trainingData;
    
    int validationLabelSize;
    int validationSampleSize;
    double* validationData;
    
    std::map<std::string, int> dictionary;
    
public:
    DataLoader();
    ~DataLoader();
    int numberOfTrainingSamples;
    int numberOfValidationSamples;
    
    // Mnist Data
    void LoadMnistTrainingData(std::string path, int sampleCount, int sampleSize, int labelSize);
    void LoadMnistValidationData(std::string path, int sampleCount, int sampleSize, int labelSize);
    
    // Sentiment Data
    void CreateDictionary(std::string path);
    void LoadDictionary(std::string path);
    void LoadSentimentTrainingData(std::string path, int sampleSize, int labelSize);
    void LoadSentimentValidationData(std::string path, int sampleSize, int labelSize);
    
    // Get Samples
    int GetRandomTrainingSample(double* sample, double* label);
    void GetRandomValidationSample(double* sample, double* label);
    void GetValidationSample(int index, double* sample, double* label);

    
};

#endif /* DataLoader_hpp */
