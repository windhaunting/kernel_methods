#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:34:22 2017

@author: fubao
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import numpy as np
import kaggle
from sklearn.kernel_ridge import KernelRidge
from commons import read_creditcard_data

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from commons import writeToFile

#kernel ridge using sklearn implementation for credit card activity dataset
                
                                
def  kernelRidgeSkLearnCV(kfold = 8, fileTestOutput = "best_cv"):
    '''
    call kernel ridge functions for different parameters for credit card activity data
    use cross validation to get out of sample out of sample meansquared error
    '''
    
    
    train_x, train_y, test_x  = read_creditcard_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)


    #train_x = normalize(train_x, axis=0)
    #test_x = normalize(test_x, axis=0)
    
    #train_x =  StandardScaler().fit_transform(train_x)
    #test_x =  StandardScaler().fit_transform(test_x)
    
    alphaParaLst = [1, 0.0001]
    
    gammaParaLst = [None, 1, 0.001]
    
    kernelParaLst = ["rbf", "polynomial", "linear"]
    
    mseErrorSmallest = 2**32
    #mseErrorLst = []
    for alpha in alphaParaLst:
        for gamma in gammaParaLst:
            for kernel in kernelParaLst:
                clf = KernelRidge(alpha=alpha, gamma = gamma, degree=3, kernel=kernel)
                mseError = -1*np.mean(cross_val_score(clf, train_x, train_y, cv=kfold, scoring="neg_mean_squared_error"))
                print ("mseError: ", alpha, gamma, kernel,  mseError)
                #mseErrorLst.append(mseError)
                if mseError < mseErrorSmallest:
                    mseErrorSmallest = mseError
                    paramtersBest = [alpha, gamma, kernel]
                
    print ("best mseError: ", kfold, paramtersBest,  mseErrorSmallest)

    #train whole data
    clf = KernelRidge(alpha=paramtersBest[0], gamma = paramtersBest[1], degree=3, kernel=paramtersBest[2])
    clf.fit(train_x, train_y)
    yPred = clf.predict(test_x)

    #output file
    if fileTestOutput != "":
        kaggle.kaggleize(yPred, fileTestOutput, True)
            
     
    
    
def trainKernelRidgeExtra(fileTestOutput, resultFile):
    '''
    for extra credit 1
    try different kernel ridge model or even different kernel
    how to select effective kernel
    '''
    
    train_x, train_y, test_x  = read_creditcard_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)
  
    #[1, 0.01, 0.001, 0.0001]
    fd = open(resultFile, 'a')

    kfoldLst = range(3,12)
    smallestError = 2**32
    
    for kfold in kfoldLst:
        parameters = {'kernel':('rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2'), 'alpha': np.linspace(0.001, 0.1, 100), 'gamma':[0.001, 1, 100]}
        
        clf = GridSearchCV(KernelRidge(), parameters, cv=kfold, n_jobs=8)   #scoring= "neg_mean_squared_error" )
        clf.fit(train_x, train_y)
        meanTestError = clf.cv_results_['mean_test_score']
        bestPara = clf.best_estimator_
        
        if clf.best_score_ < smallestError:
            smallestError = clf.best_score_
            paramtersBest = [bestPara.alpha, bestPara.gamma, bestPara.degree,  bestPara.kernel, kfold, clf.best_score_]
            
        print ("trainKernelRidgeExtra Result : ", bestPara.alpha, bestPara.gamma, bestPara.degree,  bestPara.kernel, clf.best_score_, meanTestError,)
        writeToFile(fd, [bestPara.alpha, bestPara.gamma, bestPara.degree,  bestPara.kernel, kfold, clf.best_score_] + list([meanTestError]))        
       # kwargs = {'n_neighbors': bestPara.n_neighbors}

        clf = KernelRidge(alpha=bestPara.alpha, gamma = bestPara.gamma, degree=bestPara.degree, kernel=bestPara.kernel)
        clf.fit(train_x, train_y)
        predY = clf.predict(test_x)
        
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutput != "":
            kaggle.kaggleize(predY, fileTestOutput + str(kfold), True)

    print ("best final trainKernelRidgeExtra Result: ", paramtersBest)
    