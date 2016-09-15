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
    
    // (a, b, c) x (x, y, z) = (ax, ay, az,  bx, by, bz,  cx, cy, cz)
    void TensorProduct(double* out, double* a, double* b, int size_a, int size_b)
    {
        for(int i=0; i<size_a; i++)
            for(int e=0; e<size_b; e++)
                out[(i*size_b)+e] = a[i] * b[e];
    }
    
    void LayerPropagation(double* target, double* source, double* weights, int target_size, int source_size)
    {
        for(int t=0; t<target_size; t++)
        {
            target[t] = 0.0;
            for(int s=0; s<source_size; s++)
            {
                double value = source[s] * weights[(t*source_size)+s];
                if(value != value)
                {
                    std::cout << "NAAAN\n";
                    return;
                }
                target[t] += value;
            }
        }
    }
    
    void WeightsBackpropagation(double* deltaWeights, double* source, double* weights, int source_size, int target_size)
    {
        for(int t=0; t<target_size; t++)
        {
            for(int s=0; s<source_size; s++)
                deltaWeights[(t*source_size)+s] = source[s] * weights[(t*source_size)+s];
        }
    }
    
    void LayerBackpropagation(double* deltaTarget, double* source, double* weights, int source_size, int target_size)
    {
        for(int t=0; t<target_size; t++)
        {
            deltaTarget[t] = 0;
            for(int s=0; s<source_size; s++)
                deltaTarget[t] += weights[(t*source_size)+s] * source[s];
        }
    }
}