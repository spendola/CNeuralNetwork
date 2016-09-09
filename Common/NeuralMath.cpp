//
//  NeuralMath.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "NeuralMath.hpp"

namespace neuralmath
{
    double sigmoid(double z)
    {
        return 1.0/(1.0+std::exp(-z));
    }
    
    double sigmoidprime(double z)
    {
        double sprime = sigmoid(z);
        return sprime*(1.0-sprime);
    }
    
    double tanh(double z)
    {
        double ezp = std::exp(z);
        double ezn = std::exp(-z);
        return (ezp-ezn)/(ezp+ezn);
    }
    
    double quadraticcost(double* x, double* y, int size)
    {
        double distance = 0.0;
        for (int i=0; i<size; i++)
        {
            distance += pow((x[i] - y[i]), 2.0);
        }
        return 0.5 * (pow(sqrt(distance), 2.0));
    }
    
    void softmax(double* z, int size)
    {
        double sum = 0.0;
        for(int i=0; i<size; i++)
            sum += std::exp(z[i]);
        for(int i=0; i<size; i++)
        {
            z[i] = std::exp(z[i])/sum;
            if(z[i] != z[i])
                z[i] = 0.0000001;
        }
    }
}