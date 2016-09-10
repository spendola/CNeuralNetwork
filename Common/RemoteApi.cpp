//
//  RemoteApi.cpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/9/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "RemoteApi.hpp"

RemoteApi::RemoteApi()
{
    isEnabled = LoadRemoteAddress();
}

RemoteApi::~RemoteApi()
{
    
}

void RemoteApi::PublishMessage(std::string message)
{
    if(!isEnabled) return;
    
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();
    
    message = messagePath + curl_easy_escape(curl, message.c_str(), (int)message.length());
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &RemoteApi::write_data);
        
        res = curl_easy_perform(curl);
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
        
        curl_easy_cleanup(curl);
    }
}

void RemoteApi::PublishValue(double value)
{
    if(!isEnabled) return;
    
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();

    std::string strValue = std::to_string(value);
    std::string message = graphPath + curl_easy_escape(curl, strValue.c_str(), (int)strValue.length());
    
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &RemoteApi::write_data);
        
        res = curl_easy_perform(curl);
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
        
        curl_easy_cleanup(curl);
    }
}

void RemoteApi::PublishCommand(std::string message)
{
    if(!isEnabled) return;
    
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();
    
    message = commandPath + curl_easy_escape(curl, message.c_str(), (int)message.length());
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &RemoteApi::write_data);
        
        res = curl_easy_perform(curl);
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
        
        curl_easy_cleanup(curl);
    }
}

std::string RemoteApi::FetchMessage()
{
    if(!isEnabled) return NULL;
    
    CURL *curl;
    CURLcode res;
    std::string response;
    
    curl = curl_easy_init();
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, remotePath.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        
        res = curl_easy_perform(curl);
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        response = std::string(curl_easy_strerror(res));
        curl_easy_cleanup(curl);
    }
    return response.substr(0, response.length()-8);
}


bool RemoteApi::LoadRemoteAddress()
{
    std::ifstream file ("remote.txt");
    if(file.is_open())
    {
        getline(file, messagePath);
        getline(file, graphPath);
        getline(file, commandPath);
        getline(file, remotePath);
        file.close();
        return true;
    }
    else
    {
        std::cout << "Unable to open RemoteApi configuration file\n";
        return false;
    }
    
}

size_t RemoteApi::write_data(void *contents, size_t size, size_t nmemb, void *userp)
{
    //((std::string*)userp)->append((char*)contents, size*nmemb);
    return size * nmemb;
}
