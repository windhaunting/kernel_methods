# Import python modules
import kaggle

from KRRS import KernelRidgeScratch
from BERR import BasisExpansionRidge

from plotting import plotKernelRegression

from commons import read_synthetic_data

from kernelRidgeSklearn import kernelRidgeSkLearnCV 

from svmSklearn import svmSklearnCV



if __name__== "__main__":
           
    train_x, train_y, test_x, test_y = read_synthetic_data()

    '''
    # question 1 d1 
    print ("begin to get the plotting from KRRS and BERR for question 1d1: " )
    iPolyLst = [2, 6]
    iTrigLst = [5, 10]

    YPredictLstMapKRRS, mseErrorLstKRRS = KernelRidgeScratch(iPolyLst, iTrigLst)
    
    iPolyLst = [2, 6]
    iTrigLst = [5, 10]
    YPredictLstMapBERR, mseErrorLstBEER = BasisExpansionRidge(iPolyLst, iTrigLst)
    
    YPredictLstMapDegreeAll = {**YPredictLstMapKRRS, **YPredictLstMapBERR}  
    
    #print('YPredictLstDegreeAll=', len(YPredictLstMapDegreeAll))
    plotKernelRegression(test_x, test_y, YPredictLstMapDegreeAll)
    
    '''
    '''
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

    '''
    # credit card activity dataset regression
    '''
    print ("begin to predict for credit card activity for question 1e: " )
    #kfoldLst = [5, 6, 7, 8, 9, 10]
    kfold = 8
    fileTestOutputDT = "../Predictions/CreditCard/best_cv_" + str(kfold)
    kernelRidgeSkLearnCV(kfold, fileTestOutputDT)
       
    '''

    #tumor data classification; presence/absence of tumor
    print ("begin to classify for tumor data for question 2a: " )
    kfold = 7
    fileTestOutputDT = "../Predictions/Tumor/best_cv_" + str(kfold)
    svmSklearnCV(kfold, fileTestOutputDT)
    

    #extra credit 1
    print ("extra credit 1 begin to predict for credit card activity: " )

    #trainKernelRidgeExtra()

'''

train_x, train_y, test_x, test_y = read_synthetic_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

train_x, train_y, test_x  = read_creditcard_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute MSE
test_y = np.random.rand(test_x.shape[0], train_y.shape[1])
predicted_y = np.random.rand(test_x.shape[0], train_y.shape[1])
print('DUMMY MSE=%0.4f' % compute_MSE(test_y, predicted_y))

# Output file location
file_name = '../Predictions/CreditCard/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, True)

train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute accuracy
test_y = np.random.randint(0, 2, (test_x.shape[0], 1))
predicted_y = np.random.randint(0, 2, (test_x.shape[0], 1))
print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

# Output file location
file_name = '../Predictions/Tumor/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)


'''
