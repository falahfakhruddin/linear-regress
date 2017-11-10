# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:53:13 2017

@author: falah.fakhruddin
"""

import sys
import numpy as np

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
    
def import_dataset(a, c):  
    X= a[:, 0]
    Y= a[:, 1]
    grad  = calculate_gradient(sum_array(X), sum_array(Y), sum_array_product(X,Y), sum_array_product(X,X), len(X) )
    coeff = calculate_coefficient(sum_array(X), sum_array(Y), len(X), grad)
    with open (temp[-1], 'a') as hasil:
        hasil.writelines("Hasil Perhitungan data ke- %d : \n" % c)
        hasil.writelines("Besar Gradien : %f \n" % grad)
        hasil.writelines("Besar koefisien: %f \n\n" % coeff)
    
if __name__ == '__main__':
#Importing Dataset
    temp = sys.argv

    print(temp)
    
    results = []
    list_result=[]
    i = 1
    j = 1
    
    open(temp[-1],"w").close() 
    
    for j in range (1,len(temp)):
        with open(temp[j]) as inputfile:
            for line in inputfile:
                if line == '---\n' or line == 'eof':                
                    T1 = [list(map(float, x)) for x in results]
                    import_dataset(np.array(T1), i)
                    i+=1
                    results = []
                else: 
                    results.append(line.strip().split(','))