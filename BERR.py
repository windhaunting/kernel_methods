#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:05:32 2017

@author: fubao
"""

import numpy as np

#basis expansion + kernel ridge

def BERR(trainData, testData, powerI, lambdaPara):     
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


    trainPhiX= np.apply_along_axis(basisExpanfunc, 0, trainX, 2).T
    #print ("test b: ", trainPhiX)
    print ("trainPhiX shape[0]: ", trainPhiX, type(trainPhiX), trainPhiX.shape)
    #construct \phi x, the basis expansion.
    #np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)
    
    
    
def basisExpanfunc(x, powerI) :
    phi = []
    for j in range(0, powerI+1):
        phi.append(pow(x, j))
    return phi

    
    