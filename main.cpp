//
//  main.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include <iostream>
#include "FcNetwork.hpp"
#include "CnvNetwork.hpp"


int main(int argc, const char * argv[])
{
    // insert code here...
    std::cout << "Neural Network Manager\n";
    
    FcNetwork::FcNetwork* net = new FcNetwork::FcNetwork();
    net->Start();

    delete(net);
    return 0;
}


