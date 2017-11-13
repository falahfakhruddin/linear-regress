# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:41:30 2017

@author: falah.fakhruddin
"""
import sys
import numpy as np
from regression import Reg
    
def import_dataset(a, c):  
    X= a[:, 0]
    Y= a[:, 1]
    
    Hasil = Reg(X, Y, len(Y))
    grad  = Hasil.calculate_gradient()
    coeff = Hasil.calculate_coefficient()
    
    with open (temp[2], 'a') as hasil:
        hasil.writelines("Hasil Perhitungan data ke- %d : \n" % c)
        hasil.writelines("Besar Gradien : %f \n" % grad)
        hasil.writelines("Besar koefisien: %f \n\n" % coeff)
    
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
    
    open(temp[2],"w").close() 
    
    import_dataset(dataset1, 1)
    import_dataset(dataset2, 2)

    