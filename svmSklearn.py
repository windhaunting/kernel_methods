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
from sklearn.model_selection import GridSearchCV
from commons import writeToFile


def  svmSklearnCV(kfold = 7, fileTestOutput = "best_cv"):
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
        


def trainSVMExtra(fileTestOutput, resultFile):
    '''
    for extra credit 1
    try different kernel ridge model or even different kernel
    how to select effective kernel
    '''
    
    train_x, train_y, test_x  = read_tumor_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)
  
    #[1, 0.01, 0.001, 0.0001]
    fd = open(resultFile, 'a')

    kfoldLst = range(3,12)
    biggestAccuracy = -2**32
    
    for kfold in kfoldLst:
        parameters = {'kernel':('linear', 'rbf','sigmoid', 'poly'), 'C':np.linspace(1, 10, 10), 'gamma':[0.001, 1, 100], 'degree':np.linspace(1, 10, 10)}
        
        clf = GridSearchCV(SVC(), parameters, cv=kfold, n_jobs=8)   #scoring= "neg_mean_squared_error" )
        clf.fit(train_x, train_y)
        meanTestError = clf.cv_results_['mean_test_score']
        bestPara = clf.best_estimator_
        
        if clf.best_score_ > biggestAccuracy:
            biggestAccuracy = clf.best_score_
            paramtersBest = [bestPara.C, bestPara.gamma, bestPara.degree,  bestPara.kernel, kfold, clf.best_score_]
            
        print ("trainKernelRidgeExtra Result : ", bestPara.C, bestPara.gamma, bestPara.degree,  bestPara.kernel, clf.best_score_, meanTestError,)
        writeToFile(fd, [bestPara.C, bestPara.gamma, bestPara.degree,  bestPara.kernel, kfold, clf.best_score_] + list([meanTestError]))        
       # kwargs = {'n_neighbors': bestPara.n_neighbors}

        clf = SVC(C=bestPara.C, gamma = bestPara.gamma, degree=bestPara.degree, kernel=bestPara.kernel)
        clf.fit(train_x, train_y)
        predY = clf.predict(test_x)
        
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutput != "":
            kaggle.kaggleize(predY, fileTestOutput + str(kfold), True)

    print ("best final trainKernelRidgeExtra Result: ", paramtersBest)
