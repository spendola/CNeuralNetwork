//
//  OpenCvPlot.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/20/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include "OpenCvPlot.hpp"

OpenCvPlot::OpenCvPlot()
{
    
}

OpenCvPlot::~OpenCvPlot()
{
    
}

void OpenCvPlot::SimplePlot(std::deque<double>* data, int height, int width)
{
    double* temp = new double[data->size()];
    for(int i=0; i<data->size(); i++)
        temp[i] = data->at(i);
    SimplePlot(temp, (int)data->size(), height, width);
    delete[] (temp);
}

void OpenCvPlot::SimplePlot(double* data, int points, int height, int width)
{
    // Create black empty images
    cv::Mat image = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    if(points == 0)
        return;
    
    // find min and max
    double min = 0.0;
    double max = 0.0;
    for(int i=0; i<points; i++)
    {
        max = data[i] > max ? data[i] : max;
        min = data[i] < min ? data[i] : min;
    }
    
    int ratio_y = int(height/max);
    int ratio_x = int(points/width) > 0 ? int(points/width) : 1;
    int interval = int(width/points) > 1 ? int(width/points) : 1;
    
    // Draw a line
    double prevPoint = height;
    for(int i=0; i<width; i++)
    {
        double average = 0.0;
        for(int e=0; e<ratio_x; e++)
            average += data[(i*ratio_x)+e];
        average = height - ((average/ratio_x)*(ratio_y - 1));
        cv::line(image, cv::Point( (i-1)*interval, prevPoint ), cv::Point( i*interval, average), cv::Scalar( 0, 255, 0 ),  1, 1 );
        prevPoint = average;
    }

    imshow("Image",image);
    
    cv::waitKey( 0 );
    cv::destroyWindow("Image");
    cv::waitKey(1);
    imshow("Image", image);
    
}
