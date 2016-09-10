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
#include <unistd.h>
#include "FcNetwork.hpp"
#include "CnvNetwork.hpp"
#include "RcNetwork.hpp"
#include "RemoteApi.hpp"
#include "Common/Helpers.hpp"

class NetworkManager
{
    
private:
    RemoteApi* remoteApi;
    bool publishNetworkStatus;
    bool subscribeToRemote;
    
    void MainMenu();
    void OptionsMenu();
    void MnistFcNetwork();
    void SentAnalysisFcNetwork();
    void LangModelRcNetwork();
    void ListenForRemote();
    void CleanTemporaryFiles();
    
    void Print(std::string str);
    void Publish(std::string str);
    std::vector<double> Fetch();
    
public:
    NetworkManager();
    ~NetworkManager();
    void Start();
};

#endif /* NetworkManager_hpp */
