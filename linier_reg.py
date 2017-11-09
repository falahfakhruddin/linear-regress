# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:41:30 2017

@author: falah.fakhruddin
"""
import sys
import pandas as pd 

def sum_array(array1):
    return sum(array1)
    
def sum_array_product(array1,array2):
    return sum(array1*array2)

def calculate_gradient(sum1, sum2, sum3, sum4, n):
    sxy = sum3 - (sum1*sum2/n)
    sxx = sum4 - (sum1*sum1/n)
    return sxy/sxx
    
def calculate_coefficient(sum1, sum2, n, m):
    x_mean = sum1/n
    y_mean = sum2/n
    return y_mean-(m*x_mean)
    
def import_dataset(a):  
    X= a[:, 0]
    Y= a[:, 1]
    grad  = calculate_gradient(sum_array(X), sum_array(Y), sum_array_product(X,Y), sum_array_product(X,X), len(X) )
    coeff = calculate_coefficient(sum_array(X), sum_array(Y), len(X), grad)
    print (grad)
    print (coeff)
    
if __name__ == '__main__':
#Importing Dataset
    temp = sys.argv

    print(temp)
    
    results1 = []
    results2 = []

    condition = False
    with open(temp[1]) as inputfile:
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
    
    import_dataset(dataset1)
    import_dataset(dataset2)

    