//
//  NetworkManager.hpp
//  CNeuralNetwork
//
//  Created by Sebastian Pendola on 9/7/16.
//  Copyright © 2016 Sebastian Pendola. All rights reserved.
//

#ifndef NetworkManager_hpp
#define NetworkManager_hpp

#include <stdio.h>
#include <iostream>
#include "FcNetwork.hpp"
#include "CnvNetwork.hpp"
#include "RcNetwork.hpp"
#include "Common/Helpers.hpp"

class NetworkManager
{
    
private:
    int MainMenu();
    void MnistFcNetwork();
    void SentAnalysisFcNetwork();
    void LangModelRcNetwork();
    void ListenForRemote();
    
public:
    NetworkManager();
    ~NetworkManager();
    void Start();
    

};

#endif /* NetworkManager_hpp */
