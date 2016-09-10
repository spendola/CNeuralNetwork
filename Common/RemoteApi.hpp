//
//  RemoteApi.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef RemoteApi_hpp
#define RemoteApi_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <curl/curl.h>
#include "Helpers.hpp"

class RemoteApi
{
private:
    bool isEnabled;
    std::string messagePath;
    std::string graphPath;
    std::string commandPath;
    std::string remotePath;
    bool LoadRemoteAddress();
    
public:
    RemoteApi();
    ~RemoteApi();
    void PublishMessage(std::string message);
    void PublishValue(double value);
    void PublishCommand(std::string message);
    std::string FetchMessage();
    static size_t write_data(void *buffer, size_t size, size_t nmemb, void *userp);

};

#endif /* RemoteApi_hpp */
