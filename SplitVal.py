# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:24:36 2017

@author: falah.fakhruddin
"""
from abstractmethod import  MachineLearn, NaiveB, Regression, SplitValidation

SV=SplitValidation("data4.txt",Regression(),0.6) #input data must begin with '@'
SV.runValidation()