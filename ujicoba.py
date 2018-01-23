# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:34:52 2018

@author: falah.fakhruddin
"""

from A import AbstractPreprocessing


class aselos(AbstractPreprocessing.AbstractPreprocessing):
    def testing(self):
        pass

    def predict(self):
        pass

    def training(self, A):
        print("training"+A)



lose = aselos()
print(lose.training(2))
