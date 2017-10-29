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

    trainX = trainData[0].reshape(trainData[0].shape[0], 1)
    trainY = trainData[1].reshape(trainData[1].shape[0], 1)
    
    testX = testData[0].reshape(testData[0].shape[0], 1)
    testY = testData[1].reshape(testData[1].shape[0], 1)
    
    #print ("trainX shape[0]: ",  type(trainX), trainX.shape[0], trainY.shape)


    trainPhiX= np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)          #.T
    #print ("test b: ", trainPhiX)
    #print ("trainPhiX shape[0]: ", trainPhiX, type(trainPhiX), trainPhiX.shape)
    #construct \phi x, the basis expansion.
    #np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)
   # kArr = np.empty((trainPhiX.shape[0], trainPhiX.shape[0]), dtype=np.float)        #zeros

    #for i in range(0, trainPhiX.shape[0]):
    #    for j in range(0, trainPhiX.shape[0]):
      
    print ("trainPhiX shape[0]: ",  powerI, type(trainPhiX), trainPhiX.shape, trainPhiX)
      
    kArr= np.dot(trainPhiX, trainPhiX.T) 
                
    ridgeParas = lambdaPara*np.identity(trainPhiX.shape[0], dtype=np.float)
    
    #W = np.dot(np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), trainPhiX).T, trainY)
    W = np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), np.dot(trainPhiX, trainY.T))
    #print ("kArr shape: ", type(kArr), kArr.shape, kArr[199][199], W.shape)
    
    #
    testPhiX = np.apply_along_axis(basisExpanfunc, 1, testX, powerI)
    YPred = np.dot(testPhiX, W)
    print ("YPred: ",YPred, powerI, type(YPred), YPred.shape)


    return YPred
    
    