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
    trainingData = NULL;
    validationData = NULL;
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


int DataLoader::CreateTokenizedDictionary(std::string path, bool removeUnfrequent, bool removeNonAlpha)
{
    int threshold = 1;
    int ignoredWords = 0;
    std::string line;
    std::ifstream file (path);
    std::map<std::string, int> temp;
    if(file.is_open())
    {
        std::cout << "Creating Tokenized Dictionary\n";
        file.seekg(0, std::ios::beg);
        
        while(getline(file, line))
        {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            std::stringstream ss(line.substr(2));
            std::string token;
            while(getline(ss, token, ' '))
            {
                if(helpers::ValidateWord(token))
                {
                    if (temp.find(token.c_str()) == temp.end())
                        temp[token.c_str()] = 1;
                    else
                        temp[token.c_str()] = temp[token.c_str()] + 1;
                }
                else
                    ignoredWords++;
            }
        }
        file.close();
        dictionary["unknown_token"] = 0;
        dictionary["start_token"] = 1;
        dictionary["end_token"] = 2;
        int i = (int)dictionary.size() + 1;
        
        // Remove unfrequent words
        for (std::map<std::string,int>::iterator it=temp.begin(); it!=temp.end(); ++it)
        {
            if((!removeNonAlpha || it->second > threshold) && (!removeUnfrequent || helpers::ValidateWord(it->first)))
                dictionary[it->first] = i++;
            else
                ignoredWords++;
        }
        
        dictionarySize = (int)dictionary.size();
        std::cout << dictionary.size() << " words added to dictionary, " << ignoredWords << " ignored\n";
        return (int)dictionary.size();
    }
    else
    {
        std::cout << "Unable to open file\n";
        return 0;
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

void DataLoader::LoadStopWords(std::string path)
{
    std::string line;
    std::ifstream file (path);
    std::cout << "Loading stop words\n";
    if(file.is_open())
    {
        stopwords.clear();
        while(getline(file, line))
        {
            line = line.substr(0, line.length()-1);
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            stopwords.push_back(line);
        }
    }
    else
    {
        std::cout << "Unable to open file\n";
    }
}

int DataLoader::GetValueFromDictionry(std::string word)
{
    if(dictionary.find(word.c_str()) != dictionary.end())
        return dictionary[word.c_str()];
    return dictionary["unknown_token"];
}

std::string DataLoader::GetWordFromDictionary(int value)
{
    std::map<std::string, int>::iterator it;
    for (it = dictionary.begin(); it != dictionary.end(); ++it )
        if (it->second == value)
            return it->first;
    return "*";
}

void DataLoader::PrintSentence(double* sentence, int size)
{
    for(int i=0; i<size; i++)
    {
        if(sentence[i] > 0)
            std::cout << GetWordFromDictionary(sentence[i]*dictionarySize) << " ";
    }
    std::cout << "\n";
}

void DataLoader::LoadSentimentTrainingData(std::string path, int sampleSize, int labelSize, bool ignoreStopWords)
{
    std::cout << "Loading Sentiment Analysis Training Data: " << path << "\n";
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
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            std::stringstream ss(line.substr(2));
            std::string token;
            
            // Sample Data
            while(getline(ss, token, ' '))
            {
                if(!ignoreStopWords || std::find(stopwords.begin(), stopwords.end(), token.c_str()) == stopwords.end())
                {
                    trainingData[(index*(trainingSampleSize + trainingLabelSize))+i++] = double(GetValueFromDictionry(token)) / dictionarySize;
                    if (i == (sampleSize-1))
                        break;
                }
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

void DataLoader::LoadSentimentValidationData(std::string path, int sampleSize, int labelSize, bool ignoreStopWords)
{
    std::cout << "Loading Sentiment Analysis Validation Data: " << path << "\n";
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
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            std::stringstream ss(line.substr(2));
            std::string token;
            
            // Sample Data
            while(getline(ss, token, ' '))
            {
                if(!ignoreStopWords || std::find(stopwords.begin(), stopwords.end(), token) == stopwords.end())
                {
                    validationData[(index*(validationSampleSize + validationLabelSize))+i++] = double(GetValueFromDictionry(token)) / dictionarySize;
                    if (i == (sampleSize-1))
                        break;
                }
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

void DataLoader::LoadLanguageModelTrainingData(std::string path, int sampleSize)
{
    std::cout << "Loading Language Modeling Training Data: " << path << "\n";
    numberOfTrainingSamples = 0;
    trainingSampleSize = sampleSize;
    
    std::string line;
    std::ifstream file (path);
    if(file.is_open())
    {
        std::deque<std::string> sentences;
        while(getline(file, line))
            numberOfTrainingSamples++;
        
        trainingData = new double[numberOfTrainingSamples * sampleSize]();
        
        file.clear();
        file.seekg(0, std::ios::beg);
        int index = 0;
        
        while(getline(file, line))
        {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            std::stringstream ss(line.substr(2));
            std::string token;
            
            trainingData[index*sampleSize] = dictionary["start_token"];
            
            int i = 1;
            while(getline(ss, token, ' ') && i<(sampleSize-1))
            {
                trainingData[(index*sampleSize)+i++] = GetValueFromDictionry(token);
            }
            
            trainingData[(index*sampleSize)+i] = dictionary["end_token"];
            index++;
        }
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

int DataLoader::GetLanguageTrainingSample(double* sample, double* expected)
{
    int randomSample = (rand() % numberOfTrainingSamples);
    int sampleIndex = randomSample * (trainingSampleSize);
    
    int sentenceSize = 0;
    for(int i=sampleIndex; i<(sampleIndex+trainingSampleSize); i++)
    {
        sentenceSize++;
        if(trainingData[i] == dictionary["end_token"])
            break;
    }
    
    int sentenceCut = 1 + (rand() % (std::max(sentenceSize-1, 1)));
    for(int i=0; i<sentenceCut; i++)
    {
        sample[i] = trainingData[i+sampleIndex];
        expected[i] = trainingData[i+sampleIndex];
    }
    expected[sentenceCut] = trainingData[sentenceCut+sampleIndex];
    return sentenceCut;
}

void DataLoader::GetValidationSample(int index, double* sample, double* label)
{
    int sampleIndex = index < 0 ? (rand() % numberOfValidationSamples) * (validationSampleSize+validationLabelSize) : index*(validationSampleSize+validationLabelSize);
    
    for(int i=0; i<validationSampleSize; i++)
        sample[i] = validationData[sampleIndex+i];
    for(int i=0; i<validationLabelSize; i++)
        label[i] = validationData[sampleIndex+validationSampleSize+i];
}

