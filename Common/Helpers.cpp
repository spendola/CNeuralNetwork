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
    
    int RandomInRange(double* values, int size, int range)
    {
        double max = 0.0;
        std::deque<int> maxIndex;
        for(int i=1; i<size; i++)
            if(values[i] > max)
            {
                max = values[i];
                maxIndex.push_front(i);
                if(maxIndex.size() > range)
                    maxIndex.pop_back();
            }

        if(maxIndex.size() > 0)
            return maxIndex[rand() % maxIndex.size()];
        return maxIndex[0];
    }
    
    std::vector<double> ParseInstruction(std::string instruction)
    {
        std::vector<double> output;
        std::stringstream ss(instruction);
        std::string token;
        while(getline(ss, token, ','))
            output.push_back(atof(token.c_str()));
        return output;
    }
    
    std::string GetTime()
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [80];
        
        time (&rawtime);
        timeinfo = localtime (&rawtime);
        
        strftime (buffer,80,"%F %I:%M%p",timeinfo);
        puts (buffer);
        
        return buffer;
    }
    
    double Percentage(double part, double total)
    {
        return 100.0*(part/total);
    }
    
    std::string ToString(double value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
    
    bool CheckForNan(double* data, int size)
    {
        for(int i=0; i<size; i++)
            if(data[i] != data[i])
                return true;
        return false;
    }
    
    std::string SelectFile(std::string path, std::string suffix)
    {
        int choice;
        DIR *d;
        struct dirent *dir;
        std::vector<std::string> files;
        d = opendir(path.c_str());
        if (d)
        {
            while ((dir = readdir(d)) != NULL)
            {
                std::string file = dir->d_name;
                if(file.size() >= suffix.size() && file.compare(file.size() - suffix.size(), suffix.size(), suffix) == 0)
                    files.push_back(file);
            }
            closedir(d);
        }
        
        for(int i=0; i<files.size(); i++)
            std::cout << i << " - " << files[i] << "\n";
        std::cout << "choose a file: ";
        choice = SafeCin();
        
        return path + files[choice];
    }
    
    int SafeCin()
    {
        int n;
        std::cin >> n;
        if(!std::cin)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return 0;
        }
        return n;
    }
    
}

namespace history
{
    void set(std::string str)
    {
        std::ofstream writer;
        writer.open ("../Configuration/History.txt", std::ios::app);
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
        std::ifstream file ("../Configuration/History.txt");
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
        std::ifstream file ("../Configuration/History.txt");
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
