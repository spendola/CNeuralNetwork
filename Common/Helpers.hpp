//
//  Helpers.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/21/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#define SafeDelete(p) if ((p) != NULL) { delete (p); (p) = NULL; }
#define SafeDeleteArray(p) if ((p) != NULL) { delete[] (p); (p) = NULL; }

#ifndef Helpers_hpp
#define Helpers_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <random>
#include <ctime>
#include <cmath>
#include "dirent.h"

namespace helpers
{
    void InitializeRandomArray(double* data, int size);
    void PrintArray(std::string label, double* data, int size);
    void PrintArrayEx(std::string label, double* data, int size, int precision);
    void PrintLabeledArray(std::string label, double* data, int size);
    void PrintMatrix(double* data, int width, int height);
    int RandomInRange(double* values, int size, int range);

    std::string GetTime();
    std::string ToString(double value);
    bool ValidateWord(std::string str);
    double Percentage(double part, double total);
    
    int ParseOutput(double* output, int size);
    bool ParseParameters(double* parameters, int size);
    std::vector<double> ParseInstruction(std::string instruction);
    
    bool CheckForNan(double* data, int size);
    
    std::string SelectFile(std::string path, std::string suffix);
    int SafeCin();
    
    int CalculateOutputSize(int inputWidth, int inputHeight, int featureWidth, int featureHeight, int poolingWidth, int poolingHeight);
}

namespace history
{
    void set(std::string);
    void get();
    void get(std::string);
}

#endif /* Helpers_hpp */
