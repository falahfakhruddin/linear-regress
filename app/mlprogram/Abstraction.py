# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:36:04 2018

@author: falah.fakhruddin
"""
from abc import ABC, abstractmethod


class AbstractML(ABC):
      @abstractmethod
      def training(self):
            pass
      
      @abstractmethod
      def predict(self):
            pass
      
      @abstractmethod
      def testing(self):
            pass

class AbstractPreprocessing(ABC):
      @abstractmethod
      def transform(self):
            pass
