# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:24:36 2017

@author: falah.fakhruddin
"""
import sys
temp=sys.argv
from abstractmethod import  MachineLearn, NaiveB, Regression, SplitValidation

SV=SplitValidation("data3.txt",Regression(),0.5) #input data must begin with '@'
SV.runValidation(temp[-1])