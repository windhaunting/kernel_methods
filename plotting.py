#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:06:47 2017

@author: fubao
"""

import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

def plotExample():
    #Create values and labels for bar chart
    values =np.random.rand(3)
    inds   =np.arange(3)
    labels = ["A","B","C"]
    
    #Plot a bar chart
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, values, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Method") #X-axis label
    plt.title("Error vs Method") #Plot title
    plt.xlim(-0.5,2.5) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    
    #Set the bar labels
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(labels) #label values
    
    #Save the chart
    plt.savefig("../Figures/example_bar_chart.pdf")
    
    #Create values and labels for line graphs
    values =np.random.rand(2,5)
    inds   =np.arange(5)
    labels =["Method A","Method B"]
    
    #Plot a line graph
    plt.figure(2, figsize=(6,4))      #6x4 is the aspect ratio for the plot
    plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
    plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Value") #X-axis label
    plt.title("Error vs Value") #Plot title
    plt.xlim(-0.1,4.1) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    plt.legend(labels,loc="best")
    
    #Save the chart
    plt.savefig("../Figures/example_line_plot.pdf")
    
    #Displays the plots.
    #You must close the plot window for the code following each show()
    #to continue to run
    plt.show()
    
    #Displays the charts.
    #You must close the plot window for the code following each show()
    #to continue to run
    plt.show()


def plotKernelRegression(testX, testTrueY, YPredictLstMapDegreeAll):  
    #Plot a line graph
    
    # plot with various axes scales
    plt.figure(1, figsize=(18,20))  # figsize=(6,4))
    
    # regression result
    subPlotNums = len(YPredictLstMapDegreeAll)
    methodsStr = ["KRRS ", "BERR "]
    kernelStr1 = ["Polynomial, degree = 2", "Polynomial, degree = 6"]
    kernelStr2 = ["Trignometric, degree = 5 ", "Trignometric, degree =  10 "]

    for i in range(421, subPlotNums+421):
        
        plt.subplot(i)
        plt.plot(testX, testTrueY, 'or')
        plt.plot(testX, YPredictLstMapDegreeAll[i-421], '*b')
        #plt.plot(testTrueY,'or-', linewidth=3) #Plot the first series in red with circle marker
        
        if (i-421) % 2 == 0:
            
            tmpStrInit = methodsStr[0]
        else:
            tmpStrInit = methodsStr[1]
        
        if (i-421) < 2:
            tmpStrMiddle = kernelStr1[0]
        elif (i-421) < 4:
            tmpStrMiddle = kernelStr1[1]
        elif (i-421) < 6:
            tmpStrMiddle = kernelStr2[0]
        elif (i-421) < 8:
            tmpStrMiddle = kernelStr2[1]
        
        tmpStrLast = " ,lambda = 0.1"
        #plt.yscale('linear')
        plt.xlabel("Test X")
        plt.ylabel("True/Predict Y")
        plt.title( tmpStrInit + tmpStrMiddle + tmpStrLast)
        
    plt.savefig("../Figures/Problem1d1PredictPlot.pdf")
    plt.show()


