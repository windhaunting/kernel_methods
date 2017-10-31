#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:22:03 2017

@author: fubao
"""

import numpy as np
import csv

#Commons read file


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

def 	writeToFile(fd, row):
    #fd = open(filePath, 'a')
    writer = csv.writer(fd)
    writer.writerow(row)
    

############################################################################