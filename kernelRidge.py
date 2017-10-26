#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:30:43 2017

@author: fubao
"""

#implementation of kernel ridge regression

class kernelRidge(object): 
    def __init__(self):
      pass

    
    
    def KRRS(trainData, testData, k_i):     
        '''
        kernel ridge regression from scratch
        '''
    
        trainX = trainData[0]
        trainY = trainData[1]
        
        