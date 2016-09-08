//
//  main.cpp
//  NeuralNetworkManager
//
//  Created by Sebastian Pendola on 8/8/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#include <iostream>
#include "NetworkManager.hpp"

int main(int argc, const char * argv[])
{
    NetworkManager* manager = new NetworkManager();
    manager->Start();
    
    delete(manager);
    return 0;
}


