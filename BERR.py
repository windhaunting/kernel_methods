#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:05:32 2017

@author: fubao
"""

import numpy as np

#basis expansion + kernel ridge

def BERR(trainData, testData, basisExpanfunc, powerI, lambdaPara):     
    '''
    kernel ridge regression from scratch
    k(x1,x2) = (1+x1 * x2) ^i
    input :
        synthetic data
        powerI = i
    '''

    trainX = trainData[0]
    trainY = trainData[1]
    
    testX = testData[0]
    testY = testData[1]
    
    print ("trainX shape[0]: ", trainX, type(trainX), trainX.shape[0], trainY.shape)


    trainPhiX= np.apply_along_axis(basisExpanfunc, 0, trainX, powerI).T
    #print ("test b: ", trainPhiX)
    print ("trainPhiX shape[0]: ", trainPhiX, type(trainPhiX), trainPhiX.shape)
    #construct \phi x, the basis expansion.
    #np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)
   # kArr = np.empty((trainPhiX.shape[0], trainPhiX.shape[0]), dtype=np.float)        #zeros

    #for i in range(0, trainPhiX.shape[0]):
    #    for j in range(0, trainPhiX.shape[0]):
            
    kArr= np.dot(trainPhiX, trainPhiX.T) 
                
    ridgeParas = lambdaPara*np.identity(trainPhiX.shape[0], dtype=np.float)
    
    W = np.dot(np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), trainPhiX).T, trainY)
            
    print ("kArr shape: ", type(kArr), kArr.shape, kArr[199][199], W.shape)
    
    #
    testPhiX = np.apply_along_axis(basisExpanfunc, 0, testX, powerI).T
    YPred = np.dot(testPhiX, W)
    print ("YPred: ", YPred, type(YPred), YPred.shape)



    
    