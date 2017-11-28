from __future__ import division
import sys
import numpy as np
from abc import ABC, abstractmethod
import collections
import random as rn
import json

class MachineLearn(ABC):
    #__metaclass__ = ABCMeta

    @abstractmethod
    def trainingMethod(self):
        pass
    
    @abstractmethod
    def testingMethod(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def importData(self, trainingFile):
        with open(trainingFile , 'r') as file:
            listFile = file.read().split()
            listFile.pop(0)
            file.close()
        print (listFile)
        return listFile
    
class NaiveB(MachineLearn):
    def __init__(self):        
        self.featureVectors=[]
        self.featuresDict={}
        self.featureNameList=[]
        self.labelCounts = collections.defaultdict(lambda: 0)
                 
    def trainingMethod(self,out):
        for fv in self.featureVectors:
            for counter in range(0, len(fv)-1):
                temp = 1/(self.labelCounts[fv[-1]] + len(self.featuresDict[fv[-1]][self.featureNameList[counter]]))
                self.featuresDict[fv[len(fv)-1]][self.featureNameList[counter]][fv[counter]] += temp
      
        output = open('%s-NaiveBayess.txt' %out,'w')        
        data=json.dumps(self.featuresDict, indent=4)
        output.writelines(str(data))
        output.close()                
    
    def importData (self, trainingFile):
        return super().importData(trainingFile)
        
    def getValues(self,listFile):      
        for line in listFile:
            if line[0] != '@' and line[0] != "-" :  #start of actual data
                self.featureVectors.append(line.strip().lower().split(','))
            
            elif line[0] != "-":   #feature definitions
                for x in range (0, len(line.strip().split(','))):
                    self.featureNameList.append(line.strip().split(',')[x].replace('@',''))
                    
        for fv in self.featureVectors:
            self.labelCounts[fv[len(fv)-1]] += 1 #udpate count of the label
            self.featuresDict[fv[-1]] ={}
        
        z=np.array(self.featureVectors)
        for i in range (0,len(self.featureNameList)-1):
            y=z[:,i]
            unique=[]
            used = set()
            unique = [x for x in y if x not in used and (used.add(x) or True)]
            for label in self.labelCounts:
                self.featuresDict[label][self.featureNameList[i]]={}
                for line in unique:
                    temp = 1/(self.labelCounts[label]+len(unique))
                    self.featuresDict[label][self.featureNameList[i]].update({line : temp})  

    def testingMethod(self, testFile):       
        for line in testFile: 
            if line[0] != '@' and line[0] != "-":
                vector = line.strip().lower().split(',')
                print("classifier: %s ," % self.predict(vector) + "given : %s \n" %vector[-1])

    def predict(self, featureVector):               
        probabilityPerLabel = {} 
        for label in self.labelCounts:
            tempProb = 1
            for featureValue in featureVector[:4]:
                tempProb *= self.featuresDict[label][self.featureNameList[featureVector.index(featureValue)]][featureValue]
            tempProb *= self.labelCounts[label]
            probabilityPerLabel[label]=tempProb
        print (probabilityPerLabel)
        return max(probabilityPerLabel, key = lambda classLabel: probabilityPerLabel[classLabel])
          
class Regression(MachineLearn):
    def __init__(self):
        self.trainX= []
        self.testX= []
        self.trainY= []
        self.testY= []
        self.dictGrad={}
        self.m=0
        self.b=0
        self.feature=0
        self.iteration=10000
        self.learningRate=0.01
        self.key=[]
        self.regress={}        
    
    def trainingMethod(self, out, i=None):
        self.regress["Gradien"]={}
        self.regress["Koefisien"]={}
        for x in range(0,self.feature):
            self.trainX = self.dictGrad[self.key[x]][:, 0]
            self.trainY = self.dictGrad[self.key[x]][:, 1]

            if i is None:
                self.gradientDescent(x)
                
            else:
                self.statRegression(x)
        
        output = open('%s-LinearRegression.txt' %out,'w')
        data=json.dumps(self.regress, indent=4)
        output.writelines(str(data))
        output.close()
        
    def statRegression (self, x):    
        n=len(self.trainX)
        #calculate gradient
        sum_x= sum(self.trainX)
        sum_y=sum(self.trainY)
        sum_xx=sum(self.trainX * self.trainX)
        sum_xy=sum(self.trainX * self.trainY)
        sxy = sum_xy - (sum_x * sum_y / n)
        sxx = sum_xx - (sum_x * sum_x / n)
        self.m= sxy/sxx
        self.regress['Gradien'][self.key[x]]=self.m

        #calculate coefficient
        x_mean = sum(self.trainX)/n
        y_mean = sum(self.trainY)/n
        self.b=y_mean-(self.m * x_mean)
        self.regress['Koefisien'][self.key[x]]=self.b
        
 
    def gradientDescent(self, j):
        for i in range(self.iteration):
            b_gradient = 0
            m_gradient = 0
            N = len(self.trainX)
            for i in range(0, N):
                x = self.trainX[i]
                y = self.trainY[i]
                b_gradient += -(2/N) * (y - ((self.m * x) + self.b))
                m_gradient += -(2/N) * x * (y - ((self.m * x) + self.b ))
            self.b = self.b - (self.learningRate * b_gradient)
            self.m = self.m - (self.learningRate * m_gradient)
        self.regress['Gradien'][self.key[j]]=self.m
        self.regress['Koefisien'][self.key[j]]=self.b
    
    def importData(self, trainingFile):
        return super().importData(trainingFile)
        
    def getValues(self, listFile): 
        result = []
        condition = True
   
        for line in listFile:
            if condition == True and line[0] != '@':
                self.key.append(line.strip().replace('/n',''))
                self.dictGrad[self.key[self.feature]]={}
                condition = False
            elif line[0] == '-':
                T1 = [list(map(float, x)) for x in result]
                self.dictGrad[self.key[self.feature]]=np.array(T1)
                result =[]
                condition = True
                self.feature +=1
            elif line[0] != '@':
                result.append(line.strip().split(','))
    
    def testingMethod(self, testFile):
        results1 = []
        for line in testFile:
            if line[0] != '@':
                results1.append(line.strip().split(','))
        
        T1 = [list(map(float, x)) for x in results1]
        dataset1= np.array(T1)
        self.testX= dataset1[:, 0]
        self.testY=dataset1[:,1]

        #kalkulasi 
        error =0
        for i in range(0,len(self.testY)):
            error += abs(self.testY[i]-self.predict(i))
        print('\nError :',error)
        
    def predict(self,i):
        return self.m*self.testX[i]+self.b

class SplitValidation():
    def __init__(self,validationFile,model,size=0.5):
        self.trainList = []
        self.testList = []
        self.file = validationFile
        self.type = model
        self.size = size
        self.splittingData()
        
    def splittingData(self): 
        temp=[]
        with open(self.file, "r") as f:
            data = f.read().split()
            condition = self.selectCondition(data)
            data.pop(0) #delete Title
        self.trainList.append(data[0]) #append @ line to Train List
        self.testList.append(data[0]) #append @ line to Test List
        data.pop(0)
        
        for line in data:
            if condition == True:
                self.trainList.append(line.strip())
                self.testList.append(line.strip())
                condition = False
            else:
                if line[0] != "-" :
                    temp.append(line.strip())
                else:
                    self.shuffleData(temp)
                    self.trainList.append(line.strip())
                    self.testList.append(line.strip())
                    temp=[]
                    condition = True
        print (self.trainList)
        print (self.testList)
        
                    
    def shuffleData(self, temp):
        rn.shuffle(temp)
        print (temp)
        amount=int(len(temp)*self.size)+1
        for i in range (0,len(temp)):
            if i < amount:
                self.trainList.append(temp[i])
            else:
                self.testList.append(temp[i])
                
    def selectCondition(self,data):
        if data[0] == "Linear-Regression":
            return True
        else:
            return False 
                   
    def runValidation(self,out):
        self.type.getValues(self.trainList)
        self.type.trainingMethod(out)
        self.type.testingMethod(self.testList)
  
if __name__ == "__main__":
    temp=sys.argv
    txtfile="trainset.txt"
    with open(txtfile,'r') as inputfile:
        y=inputfile.read().split()
    if y[0] == 'naive-bayess':
        model = NaiveB()
        model.getValues(txtfile)
        model.trainingMethod()
        model.testingMethod("datates.txt",temp[-1])
    else:
        Hasil = Regression()
        Hasil.getValues(txtfile)
        Hasil.trainingMethod()
        Hasil.testingMethod('data4.txt',temp[-1])
