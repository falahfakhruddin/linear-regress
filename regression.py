# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:30:16 2017

@author: falah.fakhruddin
"""
class Reg():
    def __init__(self, X, Y, n):
        self.X= X
        self.Y= Y
        self.n= n
            
    def calculate_gradient(self):
        sum_x= sum(self.X)
        sum_y=sum(self.Y)
        sum_xx=sum(self.X* self.X)
        sum_xy=sum(self.X* self.Y)
        sxy = sum_xy - (sum_x* sum_y / self.n)
        sxx = sum_xx - (sum_x* sum_x / self.n)
        return sxy/sxx
        
    def calculate_coefficient(self):
        n=self.n
        x_mean = sum(self.X)/n
        y_mean = sum(self.Y)/n
        m=self.calculate_gradient()
        return y_mean-(m*x_mean)