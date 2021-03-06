# Import python modules
import kaggle

from KRRS import KernelRidgeScratch
from BERR import BasisExpansionRidge

from plotting import plotKernelRegression

from commons import read_synthetic_data

from kernelRidgeSklearn import kernelRidgeSkLearnCV 

from kernelRidgeSklearn import trainKernelRidgeExtra

from svmSklearn import svmSklearnCV

from svmSklearn import trainSVMExtra
 

if __name__== "__main__":
           
    train_x, train_y, test_x, test_y = read_synthetic_data()

    
    # question 1 d1 
    print ("begin to get the plotting from KRRS and BERR for question 1d1: " )
    iPolyLst = [2, 6]
    iTrigLst = [5, 10]

    #KRRS
    YPredictLstMapKRRS, mseErrorLstKRRS = KernelRidgeScratch(iPolyLst, iTrigLst)
    
    
    #BERR
    YPredictLstMapBERR, mseErrorLstBEER = BasisExpansionRidge(iPolyLst, iTrigLst)
    
    YPredictLstMapDegreeAll = {**YPredictLstMapKRRS, **YPredictLstMapBERR}  
    
    #print('YPredictLstDegreeAll=', len(YPredictLstMapDegreeAll))
    plotKernelRegression(test_x, test_y, YPredictLstMapDegreeAll)
    
    
    #question 1d2
    print ("begin to get the MSE from KRRS and BERR for question 1d2: " )

    iPolyLst = [1, 2, 4, 6]      #different polynomial kernel function degrees
    iTrigLst = [3, 5, 10]       #different trignometric kernel function degrees

    YPredictLstMapKRRS, mseErrorLstKRRS = KernelRidgeScratch(iPolyLst, iTrigLst)
    
    iPolyLst = [1, 2, 4, 6]     #different polynomial basis function degrees
    iTrigLst = [3, 5, 10]       #different trignometric basis function degrees
    YPredictLstMapBERR, mseErrorLstBEER = BasisExpansionRidge(iPolyLst, iTrigLst)
    
    print ("mseErrors: ", mseErrorLstKRRS)
    print ("mseErrors: ", mseErrorLstBEER)

    
    # credit card activity dataset regression
    
    print ("begin to predict for credit card activity for question 1e: " )
    #kfoldLst = [5, 6, 7, 8, 9, 10]
    kfold = 8
    fileTestOutput = "../Predictions/CreditCard/best_cv_" + str(kfold)
    kernelRidgeSkLearnCV(kfold, fileTestOutput)
       
    

    
    #tumor data classification; presence/absence of tumor
    print ("begin to classify for tumor data for question 2a: " )
    kfold = 7
    fileTestOutput = "../Predictions/Tumor/best_cv_" + str(kfold)
    svmSklearnCV(kfold, fileTestOutput)
    
    
    #extra credit 1
    print ("extra credit 1 begin to predict for credit card activity: " )
    fileTestOutput = "../Predictions/CreditCard/best_extra_credit.csv_"
    resultFile = "../Predictions/CreditCard/resultFile.csv"
    trainKernelRidgeExtra(fileTestOutput, resultFile)

    
    
    #extra credit 2
    print ("extra credit 2 begin to classify for tumor: " )
    fileTestOutput = "../Predictions/Tumor/best_extra_credit.csv_"
    resultFile = "../Predictions/Tumor/resultFile.csv"
    trainSVMExtra(fileTestOutput, resultFile)
    
