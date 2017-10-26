#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:30:43 2017

@author: fubao
"""

#implementation of kernel ridge regression
import numpy as np



def KRRS(trainData, testData, i):     
    '''
    kernel ridge regression from scratch
    k(x1,x2) = (1+x1 * x2) ^i
    '''

    trainX = trainData[0]
    trainY = trainData[1]
    
    for xi in trainX:
        for xj in trainX:
            kij = (1 + np.dot(xi, xj))