# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:53:24 2017

@author: falah.fakhruddin
"""
import sys
from abstractmethod import MachineLearn, Regression, NaiveB

temp=sys.argv
print (temp)

for j in range (1,len(temp)-1):
    txtfile=temp[j]
    with open(txtfile,'r') as inputfile:
        y=inputfile.read().split()
    if y[0] == 'naive-bayess':
        model = NaiveB()
        trainFile = model.importData(txtfile)
        model.getValues(trainFile)
        model.trainingMethod(temp[-1])
        testFile = model.importData(txtfile)
        model.testingMethod(testFile)
    else:
        model = Regression()
        trainFile = model.importData(txtfile)
        model.getValues(trainFile)
        model.trainingMethod(temp[-1])
        testFile = model.importData(txtfile)
        model.testingMethod(testFile)
        
        
"""        
dictionary={'BRI': {'Gradien': 0.99999999997024835, 'Koefisien': 1.3309494659541973e-10}, 'PERMATA': {'Gradien': 3.976190476182961, 'Koefisien': -16.190476190400137}, 'Title': 'data3.txt', 'BTN': {'Gradien': 1.7000000007962246, 'Koefisien': 0.99999999643806037}, 'BANK MEGA': {'Gradien': 5.1999999993160451, 'Koefisien': -4.199999996940301}}
json=pickle.dumps(dictionary)
file= open('file.txt', 'w')
file.write(json)
file.close()
"""