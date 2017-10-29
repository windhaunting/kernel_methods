# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

from KRRS import KRRS
from BERR import BERR

from math import sin
from math import cos
from math import radians

from plotting import plotKernelRegression

############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
	print('Reading tumor data ...')
	train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################


def kernelFuncPoly(x1, x2, powerI):
    '''
    polynominal function
    k(x1,x2) = (1+x1 * x2) ^i
    '''
    
    return pow((1 + np.dot(x1, x2)), powerI)


def kernelFuncTrigo(x1, x2, i):
    '''
    Trigonometric function
    k(x1; x2) = 1  + sum((sin(k δ x1) × sin(k δ x2) + cos(k δ x1) × cos(k δ x2))) k =1 to i
    '''
    sigma = 0.5
    kxx = 1 + np.sum(sin(radians(k*sigma*x1)) * sin(radians(k*sigma*x2))  + cos(radians(k*sigma*x1)) * cos(radians(k*sigma*x2))  for k in range(1, i+1))
    
    return kxx


def basisExpansPoly(x, i):
    # \phi(x) = [1, x, x^2, ...., x^i]
    phi = []
    #print ("xxxxxxxxxxxxaa: ", x, len(x))
    
    for j in range(0, i+1):
        phi.append(pow(x[0], j))
    return phi

def basisExpansTrigo(x, i):
    #\phi(x) = [1, sinδx, cosδx, sin2δx, cos2δx, ..., siniδx, cosiδx]
    phi = [1]
    sigma = 0.5
    #print ("xxxxxxxxxxxx: ", x, len(x), type(x))
    for j in range(1, i+1):
        #if sin(radians(j*sigma*x[0])) != 0:
        phi.append(sin(radians(j*sigma*x[0])))
        phi.append(cos(radians(j*sigma*x[0])))
    return phi


def KernelRidgeScratch():
    '''
    call kernel ridge scratch
    '''
    
    train_x, train_y, test_x, test_y = read_synthetic_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)
        
    iPolyLst = [2, 6]   #[1, 2, 4, 6]              #different kernel function indicator
    lambdaPara = 0.1
    
    iTrigLst = [5, 10]
    
    #for kernel function 1 Polynomial order 
    YPredictLst = []
    for i in iPolyLst:
        YPred = KRRS((train_x, train_y), (test_x, test_y), kernelFuncPoly, i, lambdaPara)
        
        mseError = compute_MSE(test_y, YPred)
        YPredictLst.append(YPred)
    
    for i in iTrigLst:
        YPred = KRRS((train_x, train_y), (test_x, test_y), kernelFuncTrigo, i, lambdaPara)
        mseError = compute_MSE(test_y, YPred)
        
        YPredictLst.append(YPred)
    return YPredictLst
    

def BasisExpansionRidge():
    train_x, train_y, test_x, test_y = read_synthetic_data()
    print('Train=', train_x.shape, type(train_x))
    print('Test=', test_x.shape)

    iPolyLst = [2, 6]     #  [1, 2, 4, 6]     #different polynomial basis function indicator
    lambdaPara = 0.1
    
    iTrigLst = [5, 10]            
   
    YPredictLst = [] 
    for i in iPolyLst:
        YPred = BERR((train_x, train_y), (test_x, test_y), basisExpansPoly, i, lambdaPara)
        
        mseError = compute_MSE(test_y, YPred)
        YPredictLst.append(YPred)
        
    for i in iTrigLst:
        YPred = BERR((train_x, train_y), (test_x, test_y), basisExpansTrigo, i, lambdaPara)
        mseError = compute_MSE(test_y, YPred)
        
        YPredictLst.append(YPred)

    return YPredictLst
    



if __name__== "__main__":
           
    train_x, train_y, test_x, test_y = read_synthetic_data()

    YPredictLstKRRS = KernelRidgeScratch()
    YPredictLstBERR = BasisExpansionRidge()
    
    YPredictLstDegreeAll = YPredictLstKRRS + YPredictLstBERR
    
    print('YPredictLstDegreeAll=', len(YPredictLstDegreeAll))
    plotKernelRegression(test_x, test_y, YPredictLstDegreeAll)
    
    
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
