//
//  StopWatch.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/15/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef StopWatch_hpp
#define StopWatch_hpp

#include <stdio.h>
#include <iostream>
#include <ctime>

class StopWatch
{
public:
    StopWatch();
    ~StopWatch();
    double GetElapsed();
    
private:
    std::clock_t start;
};

#endif /* StopWatch_hpp */
