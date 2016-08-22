//
//  StopWatch.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/15/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "StopWatch.hpp"

StopWatch::StopWatch()
{
    start = std::clock();
}

double StopWatch::GetElapsed()
{
    clock_t total = clock()-start;
    start = std::clock();
    return double(total)/(CLOCKS_PER_SEC/1000.0);
}

StopWatch::~StopWatch()
{

}