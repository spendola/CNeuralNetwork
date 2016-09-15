//
//  NeuralMath.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 8/22/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef NeuralMath_hpp
#define NeuralMath_hpp

#include <stdio.h>
#include <iostream>
#include <cmath>

namespace neuralmath
{
    double sigmoid(double z);
    double sigmoidprime(double z);
    double tanh(double z);
    double fastsigmoid(double z);
    void softmax(double* z, int size);
    double quadraticcost(double* x, double* y, int size);
    
    void TensorProduct(double* out, double* a, double* b, int size_a, int size_b);
    void LayerPropagation(double* target, double* source, double* weights, int target_size, int source_size);
    void WeightsBackpropagation(double* deltaWeights, double* source, double* weights,  int source_size, int target_size);
    void LayerBackpropagation(double* deltaTarget, double* source, double* weights, int source_size, int target_size);
}

#endif /* NeuralMath_hpp */
