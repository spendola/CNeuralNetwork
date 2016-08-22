//
//  Helpers.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/21/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "Helpers.hpp"

namespace helpers
{
    void PrintArray(std::string label, double* data, int size)
    {
        std::cout << label << ": ";
        for(int i=0; i<size; i++)
            std::cout << data[i] << ", ";
        std::cout << "\n";
    }
    
    
    bool ParseParameters(double* parameters, int size)
    {
        std::cin.ignore();
        std::string str;
        std::getline (std::cin, str);
        
        int i = 0;
        char *token = std::strtok((char*)str.c_str(), ",");
        while (token != NULL)
        {
            parameters[i++] = atof(token);
            token = std::strtok(NULL, " ");
        }
        
        if(i == size)
            return true;
        return false;
    }
    
    int ParseOutput(double* output, int size)
    {
        int maxIndex = 0;
        double maxOutput = -1.0;
        for(int i=0; i<size; i++)
        {
            if(output[i] > maxOutput)
            {
                maxOutput = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    double Percentage(double part, double total)
    {
        return 100.0*(part/total);
    }
    
}