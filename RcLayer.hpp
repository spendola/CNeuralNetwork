//
//  RcLayer.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef RcLayer_hpp
#define RcLayer_hpp

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#include <stdio.h>
#include <random>
#include <ctime>
#include "deque"
#include "NeuralMath.hpp"
#include "Common/Helpers.hpp"

class RcLayer
{
private:
    double* biases;
    double* weights_in;
    double* weights_out;
    double* weights_time;
    
    double* stepActivation;
    double* stepOutput;
    
    int nNeurons;
    int nVocabulary;
    
    void InitializeWeights(double lower_bound, double upper_bound);
    
public:
    RcLayer(int neurons, int vocabularySize);
    ~RcLayer();
    
    double* FeedForward(double* in, int wordsInSentence);
    void BackPropagate(int wordsInSentence, double learningRate);
    
    int CountParameters();
    void SaveParameters(std::deque<double>* parameters);
    void LoadParameters(double* parameters, int start);
};

#endif /* RcLayer_hpp */
