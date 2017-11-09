# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:18:02 2017

@author: falah.fakhruddin
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Persamaan Garis
Y=mx+c
"""

#Importing Librariees
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
results1 = []
results2 = []
condition = False
with open('data3.txt') as inputfile:
    for line in inputfile:
        if condition == False:
            if line != '---\n':
                results1.append(line.strip().split(','))
                
            else:
                condition = True
        else:
                results2.append(line.strip().split(','))
T1 = [list(map(float, x)) for x in results1]
T2 = [list(map(float, x)) for x in results2]
dataset1= np.array(T1)
dataset2= np.array(T2)
X= dataset1[:, 0]
Y= dataset1[:, 1]

#SIgma X dan Y
sig_X =0
sig_Y =0
sig_XX =0
sig_YY =0
sig_XY =0

for i in range (0, len(X)):
    sig_X += X[i]
    sig_Y += Y[i]
    sig_XX += X[i]*X[i]
    sig_YY += Y[i]*Y[i]
    sig_XY += X[i]*Y[i]

#menghituing gradien m dan konstanta c
sxy=sig_XY-(sig_X*sig_Y/len(X))
sxx=sig_XX-(sig_X*sig_X/len(X))
m=sxy/sxx 
    
y_mean=sig_Y/len(Y)
x_mean=sig_X/len(X)
c=y_mean-(m*x_mean)

#print gradien

print ("besar gradien =", m)
print ("beasar koefisien =", c)


