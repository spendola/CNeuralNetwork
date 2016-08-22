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

void OpenCvPlot::SimplePlot(double* data)
{
    // Create black empty images
    cv::Mat image = cv::Mat::zeros( 400, 400, CV_8UC3 );
    
    // Draw a line
    cv::line( image, cv::Point( 15, 20 ), cv::Point( 70, 50), cv::Scalar( 110, 220, 0 ),  2, 8 );
    imshow("Image",image);
    
    cv::waitKey( 0 );
    cv::destroyWindow("Image");
    cv::waitKey(-1);
    imshow("Image", image);
}