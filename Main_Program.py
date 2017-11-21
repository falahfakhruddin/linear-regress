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
        model.getValues(txtfile)
        model.trainingMethod()
        model.testingMethod("datates.txt",temp[-1])
    else:
        model = Regression()
        model.getValues(txtfile)
        model.trainingMethod()
        model.testingMethod('data4.txt',temp[-1])