//
//  Helpers.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/21/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef Helpers_hpp
#define Helpers_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

namespace helpers
{
    void PrintArray(std::string label, double* data, int size);
    void PrintArrayEx(std::string label, double* data, int size, int precision);
    void PrintLabeledArray(std::string label, double* data, int size);

    std::string ToString(double value);
    double Percentage(double part, double total);
    
    int ParseOutput(double* output, int size);
    bool ParseParameters(double* parameters, int size);
    std::vector<double> ParseInstruction(std::string instruction);
    
    bool CheckForNan(double* data, int size);
}

namespace history
{
    void set(std::string);
    void get();
    void get(std::string);
}

#endif /* Helpers_hpp */
