# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:53:13 2017

@author: falah.fakhruddin
"""

import sys
import numpy as np
from regression import Reg
    
def import_dataset(a, b, c):  
    X= a[:, 0]
    Y= a[:, 1]

    Hasil = Reg(X, Y, len(Y))
    grad  = Hasil.calculate_gradient()
    coeff = Hasil.calculate_coefficient()

    with open (temp[-1]+'-dataset %i.txt'%b, 'w') as hasil:
        hasil.writelines("Hasil Perhitungan dataset %s" % b)
        hasil.writelines("Besar Gradien : %f \n" % grad)
        hasil.writelines("Besar koefisien: %f \n\n" % coeff)
        hasil.close()
    
if __name__ == '__main__':
#Importing Dataset
    temp = sys.argv

    print(temp)
    
    results = []
    condition = True
    i = 1
    j = 1
    
    open(temp[-1],"w").close() 
    
    for j in range (1,len(temp)):
        with open(temp[j]) as inputfile:
            for line in inputfile:
                                
                if condition == True:
                   nama=line
                   condition = False
               
                elif line == '---\n' or line == 'eof': 
                    T1 = [list(map(float, x)) for x in results]
                    import_dataset(np.array(T1),nama, i)
                    i+=1
                    results = []
                    condition = True
                    
                else: 
                    results.append(line.strip().split(','))


