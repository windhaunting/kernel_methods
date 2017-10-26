#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:30:43 2017

@author: fubao
"""

#implementation of kernel ridge regression
import numpy as np



def KRRS(trainData, testData, i, lambdaPara):     
    '''
    kernel ridge regression from scratch
    k(x1,x2) = (1+x1 * x2) ^i
    input :
        synthetic data
    '''

    trainX = trainData[0]
    trainY = trainData[1]
    
    
    testX = testData[0]
    testY = testData[1]
    
    print ("trainX shape[0]: ", trainX.shape[0], trainY.shape)
    
    #get alpha below

    kArr = np.empty((trainX.shape[0], trainX.shape[0]), dtype=np.float)
    print ("kArr shape original: ", kArr.shape)

    #kArr = np.empty(1)
    for i in range(0, trainX.shape[0]):
        for j in range(0, trainX.shape[0]):
            xi = trainX[i]
            xj = trainX[j]
            kij = (1 + np.dot(xi, xj))
            
            #print ("kij: ", kij)
            #kArr = np.vstack((kArr, np.array(kij)))
            #print ("trainX kij: ", kArr)
            np.append(kArr, kij)
            #print ("kij: ", kArr)
    print ("kArr shape: ", type(kArr), kArr.shape)
    #print ("kArr shape: ", kArr[0][0], kArr[2][0], kArr[199][199], type(kArr), kArr.shape)
    print ("kij: ", kArr)
    
    #get
    ridgeParas = lambdaPara*np.identity(trainX.shape[0], dtype=np.float)
    
    alpha = np.dot(np.linalg.inv(kArr + ridgeParas), trainY)          #alpha for kernel ridge $\alpha = (\Phi(X)\phi^T(X)+\lambda I)^{-1}Y$ 
    
    print ("ridgeParas: ", ridgeParas, alpha, alpha.shape)
    
    
    for i in range(0, testX.shape[0]):
        
        xnew = testX[i]
        print ("kij: ", kArr)
