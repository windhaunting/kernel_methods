#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:05:32 2017

@author: fubao
"""

import numpy as np
from sklearn.linear_model import Ridge

from math import sin
from math import cos

from commons import read_synthetic_data
from commons import compute_MSE

#basis expansion + kernel ridge

def BERRScratch(trainData, testData, basisExpanfunc, powerI, lambdaPara):     
    '''
    Basis Expansion + ridge regression 
    e.g. k(x1,x2) = (1+x1 * x2) ^i
    input :
        synthetic data
        powerI = i
    '''

    trainX = trainData[0].reshape(trainData[0].shape[0], 1)
    trainY = trainData[1].reshape(trainData[1].shape[0], 1)
    
    testX = testData[0].reshape(testData[0].shape[0], 1)
    #testY = testData[1].reshape(testData[1].shape[0], 1)
    
    #print ("trainX shape[0]: ",  type(trainX), trainX.shape[0], trainY.shape)


    trainPhiX= np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)          #.T
    #print ("test b: ", trainPhiX)
    #print ("trainPhiX shape[0]: ", trainPhiX, type(trainPhiX), trainPhiX.shape)
    #construct \phi x, the basis expansion.
    #np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)
   # kArr = np.empty((trainPhiX.shape[0], trainPhiX.shape[0]), dtype=np.float)        #zeros

    #for i in range(0, trainPhiX.shape[0]):
    #    for j in range(0, trainPhiX.shape[0]):
      
    #print ("trainPhiX shape[0]: ",  powerI, type(trainPhiX), trainPhiX.shape, trainPhiX)
      
    kArr= np.dot(trainPhiX, trainPhiX.T) 
                
    ridgeParas = lambdaPara*np.identity(trainPhiX.shape[0], dtype=np.float)
    
    W = np.dot(np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), trainPhiX).T, trainY)
    #W = np.dot(np.linalg.inv(np.add(kArr, ridgeParas)), np.dot(trainPhiX.T, trainY))
    #print ("kArr shape: ", type(kArr), kArr.shape, kArr[199][199], W.shape)
    
    #
    testPhiX = np.apply_along_axis(basisExpanfunc, 1, testX, powerI)
    YPred = np.dot(testPhiX, W)
    #print ("YPred: ",YPred, powerI, type(YPred), YPred.shape)


    return YPred
    
  
def BERRRidge(trainData, testData, basisExpanfunc, powerI, lambdaPara): 
    '''
    basis expansion + sklearn ridge regression 
    '''    
    trainX = trainData[0].reshape(trainData[0].shape[0], 1)
    trainY = trainData[1].reshape(trainData[1].shape[0], 1)
    
    testX = testData[0].reshape(testData[0].shape[0], 1)
    #testY = testData[1].reshape(testData[1].shape[0], 1)
    
    trainPhiX= np.apply_along_axis(basisExpanfunc, 1, trainX, powerI)          #.T
    testPhiX = np.apply_along_axis(basisExpanfunc, 1, testX, powerI)

    #print ("trainPhiX shape[0]: ",  powerI, type(trainPhiX), trainPhiX.shape, trainPhiX)

    clf = Ridge(alpha=lambdaPara)
    clf.fit(trainPhiX, trainY)
    
    YPred = clf.predict(testPhiX)
    
    #print ("YPred: ",YPred, powerI, type(YPred), YPred.shape)

    return YPred


def basisExpansPoly(x, i):
    '''
    # \phi(x) = [1, x, x^2, ...., x^i]
    '''
    phi = []
    print ("xxxxxxxxxxxxaa: ", x, len(x), x[0])
    
    for j in range(0, i+1):
        phi.append(pow(x[0], j))
    return phi

def basisExpansTrigo(x, i):
    '''
    #\phi(x) = [1, sinδx, cosδx, sin2δx, cos2δx, ..., siniδx, cosiδx]
    '''
    phi = [1]
    sigma = 0.5
    #print ("xxxxxxxxxxxx: ", x, len(x), type(x))
    for j in range(1, i+1):
        #if sin(radians(j*sigma*x[0])) != 0:
        phi.append(sin(j*sigma*x[0]))           #radians()
        phi.append(cos(j*sigma*x[0]))
    return phi




def BasisExpansionRidge(iPolyLst, iTrigLst):
    '''
    BERR execution for plotting
    '''
    train_x, train_y, test_x, test_y = read_synthetic_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)

    lambdaPara = 0.1
   
    YPredictLstMap = {}
    indexPlot = 1
    mseErrorLst = []

    for i in iPolyLst:  #[1:]:            #test only
        #YPred = BERRScratch((train_x, train_y), (test_x, test_y), basisExpansPoly, i, lambdaPara)        
        YPred = BERRRidge((train_x, train_y), (test_x, test_y), basisExpansPoly, i, lambdaPara)

        mseError = compute_MSE(test_y, YPred)
        mseErrorLst.append(mseError)

        #print('BEER mseError poly i=', mseError, i)

        YPredictLstMap[indexPlot] = YPred
        indexPlot += 2
    
    
    for j in iTrigLst:
        #YPred = BERRScratch((train_x, train_y), (test_x, test_y), basisExpansTrigo, j, lambdaPara)
        YPred = BERRRidge((train_x, train_y), (test_x, test_y), basisExpansTrigo, j, lambdaPara)

        mseError = compute_MSE(test_y, YPred)
        mseErrorLst.append(mseError)

        #print('BEER mseError trignometric i=', mseError, j)
        YPredictLstMap[indexPlot] = YPred
        indexPlot += 2
    
    return YPredictLstMap, mseErrorLst
