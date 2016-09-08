//
//  OpenCvPlot.hpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/20/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef OpenCvPlot_hpp
#define OpenCvPlot_hpp

#include <stdio.h>
#include <deque>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

class OpenCvPlot
{
private:
    
    
public:
    OpenCvPlot();
    ~OpenCvPlot();
    void SimplePlot(std::deque<double>* data, int height, int width);
    void SimplePlot(double* data, int points, int heigth, int width);
    
};

#endif /* OpenCvPlot_hpp */
