#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:30:43 2017

@author: fubao
"""

from sklearn.preprocessing import normalize

#implementation of kernel ridge regression from scratch
import numpy as np

from math import sin
from math import cos
from math import radians


'''
def kernelFunc1(x1, x2, powerI):
    
    #k(x1,x2) = (1+x1 * x2) ^i
    
    return pow((1 + np.dot(x1, x2)), powerI)


def kernelFunc2(x1, x2, i):
    
    #k(x1; x2) = 1  + sum((sin(k δ x1) × sin(k δ x2) + cos(k δ x1) × cos(k δ x2))) k =1 to i
    
    sigma = 0.5
    kxx = 1 + np.sum(sin(radians(k*sigma*x1)) * sin(radians(k*sigma*x2))  + cos(radians(k*sigma*x1)) * cos(radians(k*sigma*x2))  for k in range(1, i+1))
    
    return kxx

'''

def KRRS(trainData, testData, kernelFunc, powerI, lambdaPara):     
    '''
    kernel ridge regression from scratch
    for different kernel function
    input :
        synthetic data
        powerI = i
        
    '''

    trainX = trainData[0]
    trainY = trainData[1]
    
    testX = testData[0]
    testY = testData[1]
    
    #trainX -= .5
    #testX  -= .5
    
    #trainX = normalize(trainX.reshape(-1,1), axis=0)
    #testX = normalize(testX.reshape(-1,1), axis=0)

    #print ("trainX shape[0]: ", trainX, trainX.shape[0], trainY.shape)
    
    #get alpha below
    kArr = np.empty((trainX.shape[0], trainX.shape[0]), dtype=np.float)        #zeros
    print ("kArr shape original: ", kArr.shape)

    #kArr = np.empty(1)
    for i in range(0, trainX.shape[0]):
        for j in range(0, trainX.shape[0]):
            xi = trainX[i]
            xj = trainX[j]
            print ()
            kij =  kernelFunc(xi, xj, powerI)  # pow((1.0 + np.dot(xi, xj)), powerI) #xi*xj) #
            
            #print ("xi, xj: ", xi, xj)
            #kArr = np.vstack((kArr, np.array(kij)))
            #print ("trainX kij: ", kij)
            kArr[i][j] = kij
            #print ("kij: ", kArr)
    
  
    #print ("kArr shape: ", type(kArr), kArr.shape, kArr[199][199])
    #print ("kArr shape: ", kArr[0][0], kArr[2][0], kArr[199][199], type(kArr), kArr.shape)
    #print ("kij: ", kArr)
       
    #get
    ridgeParas = lambdaPara*np.identity(trainX.shape[0], dtype=np.float)
    
    alpha = np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), trainY)          #alpha for kernel ridge $\alpha = (\Phi(X)\phi^T(X)+\lambda I)^{-1}Y$ 
    #print ("ridgeParas: ", ridgeParas,np.linalg.inv(np.add(kArr, ridgeParas)),  alpha, alpha.shape)
    
    YPred = np.empty((testX.shape[0]), dtype=np.float)        #zeros
    for testInd in range(0, testX.shape[0]):
        
        xnew = testX[testInd]
        #for i in range(0, trainX.shape[0]):   # $y_{new} = \sum_{i}  \alpha_i \Phi(x_i) \Phi(x_{new}) 
        #innerVal = 
        YPred[testInd] = np.sum([np.dot(alpha[i],  kernelFunc(trainX[i], xnew, powerI)) for i in range(0, trainX.shape[0])])          #sum ??

        #alpha[i]
        #print ("xnew: ", xnew, YPred[testInd])
    
    
    print ("YPred: ", type(YPred), YPred.shape)

    return YPred