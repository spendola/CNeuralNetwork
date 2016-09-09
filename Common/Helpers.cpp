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
    
    void PrintArrayEx(std::string label, double* data, int size, int precision)
    {
        std::cout << label << ": ";
        for(int i=0; i<size; i++)
            std::cout << std::setprecision(precision) << data[i] << ", ";
        std::cout << "\n";
    }
    
    void PrintLabeledArray(std::string label, double* data, int size)
    {
        std::cout << label << ": ";
        for(int i=0; i<size; i++)
            std::cout << i << ":" << data[i] << ", ";
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
        {
            history::set(str);
            return true;
        }
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
    
    bool CheckForNan(double* data, int size)
    {
        for(int i=0; i<size; i++)
            if(data[i] != data[i])
                return true;
        return false;
    }
    
}

namespace history
{
    void set(std::string str)
    {
        std::ofstream writer;
        writer.open ("Saved/History.txt", std::ios::app);
        if(writer.is_open())
        {
            writer << str;
            writer << "\n";
        }
        writer.close();
    }
    
    void get()
    {
        std::string line;
        std::ifstream file ("Saved/History.txt");
        if(file.is_open())
        {
            while(getline(file, line))
            {
                std::cout << line << "\n";
            }
            file.close();
        }
        else
        {
            std::cout << "Unable to open file\n";
        }
    }
    
    void get(std::string str)
    {
        std::string line;
        std::ifstream file ("Saved/History.txt");
        if(file.is_open())
        {
            while(getline(file, line))
            {
                if(line.substr(0, str.length()) == str)
                    std::cout << line << std::endl;
            }
            file.close();
        }
        else
        {
            std::cout << "Unable to open file\n";
        }
    }
}

namespace remote
{
    void PublishMessage(std::string message)
    {
        CURL *curl;
        CURLcode res;
        curl = curl_easy_init();
        
        std::tuple<std::string, std::string, std::string> remote = LoadRemoteAddress();
        message = std::get<0>(remote) + curl_easy_escape(curl, message.c_str(), (int)message.length());
        if(curl)
        {
            curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
            /* example.com is redirected, so we tell libcurl to follow redirection */
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            
            /* Perform the request, res will get the return code */
            res = curl_easy_perform(curl);
            /* Check for errors */
            //if(res != CURLE_OK)
            //fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
            
            /* always cleanup */
            curl_easy_cleanup(curl);
        }
    }
    
    void PublishValue(double value)
    {
        CURL *curl;
        CURLcode res;
        curl = curl_easy_init();
        
        std::tuple<std::string, std::string, std::string> remote = LoadRemoteAddress();
        std::string strValue = std::to_string(value);
        std::string message = std::get<1>(remote) + curl_easy_escape(curl, strValue.c_str(), (int)strValue.length());
        
        if(curl)
        {
            curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
            /* example.com is redirected, so we tell libcurl to follow redirection */
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            
            /* Perform the request, res will get the return code */
            res = curl_easy_perform(curl);
            /* Check for errors */
            if(res != CURLE_OK)
                fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
            
            /* always cleanup */
            curl_easy_cleanup(curl);
        }
    }
    
    void PublishCommand(std::string message)
    {
        CURL *curl;
        CURLcode res;
        curl = curl_easy_init();
        
        std::tuple<std::string, std::string, std::string> remote = LoadRemoteAddress();
        message = std::get<2>(remote) + curl_easy_escape(curl, message.c_str(), (int)message.length());
        if(curl)
        {
            curl_easy_setopt(curl, CURLOPT_URL, message.c_str());
            /* example.com is redirected, so we tell libcurl to follow redirection */
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            
            /* Perform the request, res will get the return code */
            res = curl_easy_perform(curl);
            /* Check for errors */
            //if(res != CURLE_OK)
            //fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
            
            /* always cleanup */
            curl_easy_cleanup(curl);
        }
    }
    
    std::string FetchMessage(std::string remote)
    {
        CURL *curl;
        CURLcode res;
        
        curl = curl_easy_init();
        if(curl)
        {
            curl_easy_setopt(curl, CURLOPT_URL, remote.c_str());
            /* example.com is redirected, so we tell libcurl to follow redirection */
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            
            /* Perform the request, res will get the return code */
            res = curl_easy_perform(curl);
            /* Check for errors */
            if(res != CURLE_OK)
                fprintf(stderr, "curl_easy_perform() failed: %s\n",curl_easy_strerror(res));
            
            /* always cleanup */
            curl_easy_cleanup(curl);
            
            return std::string(curl_easy_strerror(res));
        }
        return NULL;
    }
    
    std::tuple<std::string, std::string, std::string> LoadRemoteAddress()
    {
        std::string messagePath;
        std::string valuePath;
        std::string commandPath;
        std::ifstream file ("remote.txt");
        if(file.is_open())
        {
            getline(file, messagePath);
            getline(file, valuePath);
            getline(file, commandPath);
            file.close();
        }
        else
        {
            std::cout << "Unable to open file\n";
        }
        
        return std::make_tuple(messagePath, valuePath, commandPath);
    }
    
    size_t write_data(void *buffer, size_t size, size_t nmemb, void *userp)
    {
        return size * nmemb;
    }
    
}