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
#include <cmath>

namespace neuralmath
{
    double sigmoid(double z);
    double sigmoidprime(double z);
    double tanh(double z);
    double fastsigmoid(double z);
    void softmax(double* z, int size);
    
    double quadraticcost(double* x, double* y, int size);
}

#endif /* NeuralMath_hpp */
