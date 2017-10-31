#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:34:12 2017

@author: fubao
"""
import numpy as np
import kaggle
from sklearn.svm import SVC
from commons import read_tumor_data
from sklearn.model_selection import cross_val_score
 
def  svmSklearnCV(kfold = 5, fileTestOutput = "best_cv"):
    '''
    call svm train and predict for different parameters for tumor data
    use cross validation to get out of sample out of sample meansquared error
    '''
    
    train_x, train_y, test_x  = read_tumor_data()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)


    cLst = [1, 0.01, 0.0001]
    gammaLst = [1, 0.01, 0.001]
    kernelParaLst = ['rbf', 'poly=3', 'poly=5', 'linear']
    degree = 3
    
    accuracyLargest = -2**32
    for c in cLst:
        for gamma in gammaLst:
            for kernel in kernelParaLst:
                if kernel.split("=")[0] == "poly":
                    degree = int(kernel.split("=")[1])
                    kernel = kernel.split("=")[0]
                clf = SVC(C=c, kernel=kernel, degree=degree, gamma = gamma)
                
                accuracy = np.mean(cross_val_score(clf, train_x, train_y, cv=kfold, scoring="accuracy"))
                print ("accuracy: ", c, gamma, kernel, degree, accuracy)
                if accuracy > accuracyLargest:
                    accuracyLargest = accuracy
                    paramtersBest = [c, gamma, degree, kernel]
                
                
                
    print ("best accuracy parameters: ", kfold, paramtersBest,  accuracyLargest)

    #train whole data
    clf = SVC(C=paramtersBest[0], gamma = paramtersBest[1], degree=paramtersBest[2], kernel=paramtersBest[3])
    clf.fit(train_x, train_y)
    yPred = clf.predict(test_x)

    #output file
    if fileTestOutput != "":
        kaggle.kaggleize(yPred, fileTestOutput, False)
        


