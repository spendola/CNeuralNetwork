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
#include <deque>
#include "Helpers.hpp"

class DataLoader
{
private:

    double* trainingData;
    double* validationData;
    
    std::vector<std::string> stopwords;
    std::map<std::string, int> dictionary;
    
public:
    DataLoader();
    ~DataLoader();
    int numberOfTrainingSamples;
    int numberOfValidationSamples;
    int dictionarySize;
    int trainingLabelSize;
    int trainingSampleSize;
    int validationLabelSize;
    int validationSampleSize;
    
    // Mnist Data
    void LoadMnistTrainingData(std::string path, int sampleCount, int sampleSize, int labelSize);
    void LoadMnistValidationData(std::string path, int sampleCount, int sampleSize, int labelSize);
    
    // Sentiment Data
    
    int CreateDictionary(std::string path);
    int CreateTokenizedDictionary(std::string path, bool removeUnfrequent, bool removeNonAlpha);
    void LoadStopWords(std::string path);
    void LoadDictionary(std::string path);
    int GetValueFromDictionry(std::string word);
    std::string GetWordFromDictionary(int value);
    void PrintSentence(double* sentence, int size);
    
    void LoadSentimentTrainingData(std::string path, int sampleSize, int labelSize, bool ignoreStopWords);
    void LoadSentimentValidationData(std::string path, int sampleSize, int labelSize, bool ignoreStopWords);
    
    // LanguageModeling Data
    void LoadLanguageModelTrainingData(std::string path, int sampleSize);
    
    // Get Samples
    int GetRandomTrainingSample(double* sample, double* label);
    void GetValidationSample(int index, double* sample, double* label);
    int GetLanguageTrainingSample(double* sample, double* expected);
};

#endif /* DataLoader_hpp */
