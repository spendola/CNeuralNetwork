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
#include <curl/curl.h>

namespace helpers
{
    void PrintArray(std::string label, double* data, int size);
    void PrintArrayEx(std::string label, double* data, int size, int precision);
    void PrintLabeledArray(std::string label, double* data, int size);
    bool ParseParameters(double* parameters, int size);
    std::vector<double> ParseInstruction(std::string instruction);
    int ParseOutput(double* output, int size);
    double Percentage(double part, double total);
    bool CheckForNan(double* data, int size);
}

namespace history
{
    void set(std::string);
    void get();
    void get(std::string);
}

#endif /* Helpers_hpp */
