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
#include <iomanip>
#include <curl/curl.h>

namespace helpers
{
    void PrintArray(std::string label, double* data, int size);
    void PrintArrayEx(std::string label, double* data, int size, int precision);
    void PrintLabeledArray(std::string label, double* data, int size);
    bool ParseParameters(double* parameters, int size);
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

namespace remote
{
    void PublishMessage(std::string message);
    void PublishValue(double value);
    void PublishCommand(std::string message);
    std::string FetchMessage(std::string remote);
    std::tuple<std::string, std::string, std::string> LoadRemoteAddress();
    size_t write_data(void *buffer, size_t size, size_t nmemb, void *userp);
}

#endif /* Helpers_hpp */
