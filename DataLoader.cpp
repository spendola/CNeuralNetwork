//
//  DataLoader.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "DataLoader.hpp"
#include "unistd.h"

DataLoader::DataLoader()
{
    std::cout << "Initializing DataLoader\n";
    srand((unsigned)time(0));
}

DataLoader::~DataLoader()
{
    std::cout << "Cleaning up DataLoader\n";
    SafeDeleteArray(trainingData);
    SafeDeleteArray(validationData);
}


void DataLoader::LoadMnistValidationData(std::string path, int sampleCount, int sampleSize, int labelSize)
{
    int i=0;
    numberOfValidationSamples = sampleCount;
    validationSampleSize = sampleSize;
    validationLabelSize = labelSize;
    int expectedValues = numberOfValidationSamples*(validationSampleSize+validationLabelSize);
    
    std::cout << "Loading Validation Data\n";
    std::ifstream file (path, std::ios::binary);
    if(file.is_open())
    {
        validationData = new double[expectedValues];
        char buffer[sizeof(double)];
        while(file.read(buffer, sizeof(double)))
            validationData[i++] = *((double*)buffer);
        
        if(i != expectedValues)
            std::cout << "Expected number of values is different from actual number of values loaded\n";
        
        file.close();
    }
    else
    {
        std::cout << "Unable to open file\n";
    }
}

void DataLoader::LoadMnistTrainingData(std::string path, int sampleCount, int sampleSize, int labelSize)
{
    int i=0;
    numberOfTrainingSamples = sampleCount;
    trainingSampleSize = sampleSize;
    trainingLabelSize = labelSize;
    int expectedValues = numberOfTrainingSamples*(trainingSampleSize+trainingLabelSize);
    
    std::cout << "Loading Training Data\n";
    std::ifstream file (path, std::ios::binary);
    if(file.is_open())
    {
        trainingData = new double[expectedValues];
        char buffer[sizeof(double)];
        while(file.read(buffer, sizeof(double)))
            trainingData[i++] = *((double*)buffer);
        
        if(i != expectedValues)
            std::cout << "Expected number of values is different from actual number of values loaded\n";
        
        file.close();
    }
    else
    {
        std::cout << "Unable to open file\n";
    }
}



void DataLoader::CreateDictionary(std::string path)
{
    std::string line;
    std::ifstream file (path);
    if(file.is_open())
    {
        std::cout << "Creating Dictionary\n";
        file.seekg(0, std::ios::beg);
        
        int i = 1;
        while(getline(file, line))
        {
            std::stringstream ss(line.substr(2));
            std::string token;
            while(getline(ss, token, ' '))
            {
                if ( dictionary.find(token.c_str()) == dictionary.end() )
                    dictionary[token.c_str()] = i++;
            }
        }
        file.close();
        std::cout << dictionary.size() << " words added to dictionary\n";
    }
    else
    {
        std::cout << "Unable to open file\n";
    }
}

void DataLoader::LoadDictionary(std::string path)
{
    std::string line;
    std::ifstream file (path);
    if(file.is_open())
    {
        dictionary.clear();
        while(getline(file, line))
        {
            std::cout << line << std::endl;
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file\n";
    }
}

void DataLoader::LoadSentimentTrainingData(std::string path, int sampleSize, int labelSize)
{
    std::cout << "Loading Training Data: " << path << "\n";
    trainingSampleSize = sampleSize;
    trainingLabelSize = labelSize;
    numberOfTrainingSamples = 0;
    
    std::string line;
    std::ifstream file (path);
    if(file.is_open())
    {
        while(getline(file, line))
            numberOfTrainingSamples++;
        
        int size = numberOfTrainingSamples * (trainingSampleSize + trainingLabelSize);
        trainingData = new double[size];
        int index = 0;
        
        file.clear();
        file.seekg(0, std::ios::beg);
        
        while(getline(file, line))
        {
            int i = 0;
            std::stringstream ss(line.substr(2));
            std::string token;
            
            // Sample Data
            while(getline(ss, token, ' '))
            {
                if(dictionary.find(token.c_str()) == dictionary.end())
                    trainingData[(index*(trainingSampleSize + trainingLabelSize)) + i] = 0.001;
                else
                    trainingData[(index*(trainingSampleSize + trainingLabelSize)) + i] = double(dictionary[token] / 1000.0);
                //*((double*)buffer);
                i++;
                if (i == (sampleSize-1))
                    break;
            }
            
            // Label Data
            if(line[0] == '0')
                trainingData[(index*(trainingSampleSize + trainingLabelSize)) + sampleSize] = 1.0;
            else
                trainingData[(index*(trainingSampleSize + trainingLabelSize)) + sampleSize + 1] = 1.0;
            
            index++;
        }
        
        
        if(index != numberOfTrainingSamples)
            std::cout << "Expected number of values is different from actual number of values loaded\n";
        else
            std::cout << numberOfTrainingSamples << " samples added to training data\n";
            
        file.close();
    }
    else
    {
        std::cout << "Unable to open file " << path << "\n";
    }
}

void DataLoader::LoadSentimentValidationData(std::string path, int sampleSize, int labelSize)
{
    std::cout << "Loading Validation Data: " << path << "\n";
    validationSampleSize = sampleSize;
    validationLabelSize = labelSize;
    numberOfValidationSamples = 0;
    
    std::string line;
    std::ifstream file (path);
    if(file.is_open())
    {
        while(getline(file, line))
            numberOfValidationSamples++;
        
        validationData = new double[numberOfValidationSamples * (validationSampleSize + validationLabelSize)]();
        int index = 0;
        
        file.clear();
        file.seekg(0, std::ios::beg);
        
        while(getline(file, line))
        {
            int i = 0;
            std::stringstream ss(line.substr(2));
            std::string token;
            
            // Sample Data
            while(getline(ss, token, ' '))
            {
                if(dictionary.find(token) == dictionary.end())
                    validationData[(index*(validationSampleSize + validationLabelSize)) + (i++)] = 0.001;
                else
                    validationData[(index*(validationSampleSize + validationLabelSize)) + (i++)] = double(dictionary[token] / 1000.0);
                
                if (i == (sampleSize-1))
                    break;
            }
            
            // Label Data
            if(line[0] == '0')
                validationData[(index*(validationSampleSize + validationLabelSize)) + sampleSize] = 0.0;
            else
                validationData[(index*(validationSampleSize + validationLabelSize)) + sampleSize] = 1.0;
            
            index++;
        }
        
        if(index != numberOfValidationSamples)
            std::cout << "Expected number of values is different from actual number of values loaded\n";
        else
            std::cout << numberOfValidationSamples << " samples added to validation data\n";
        file.close();
    }
    else
    {
        std::cout << "Unable to open file " << path << "\n";
    }

}

int DataLoader::GetRandomTrainingSample(double* sample, double* label)
{
    int randomSample = (rand() % numberOfTrainingSamples);
    int sampleIndex = randomSample * (trainingSampleSize+trainingLabelSize);
    
    for (int i=0; i<trainingSampleSize; i++)
        sample[i] = trainingData[sampleIndex+i];
    for (int i=0; i<trainingLabelSize; i++)
        label[i] = trainingData[sampleIndex+trainingSampleSize+i];
    
    return randomSample;
}

void DataLoader::GetRandomValidationSample(double* sample, double* label)
{
    int randomSample = (rand() % numberOfValidationSamples);
    GetValidationSample(randomSample, sample, label);
}

void DataLoader::GetValidationSample(int index, double* sample, double* label)
{
    int sampleIndex = index < 0 ? (rand() % numberOfValidationSamples) * (validationSampleSize+validationLabelSize)
                        : index*(validationSampleSize+validationLabelSize);
    
    for(int i=0; i<validationSampleSize; i++)
        sample[i] = validationData[sampleIndex+i];
    for(int i=0; i<validationLabelSize; i++)
        label[i] = validationData[sampleIndex+validationSampleSize+i];
}

