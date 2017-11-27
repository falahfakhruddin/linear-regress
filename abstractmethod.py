from __future__ import division
import sys
import numpy as np
from abc import ABCMeta, abstractmethod
import collections
import random as rn
import json

class MachineLearn(object):
    __metaclass__ = ABCMeta

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
    def getValues(self):
        pass
    
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
        
    def getValues(self,trainingFile):
        self.featuresDict['title']=trainingFile
        file = open(trainingFile, 'r')
        for line in file:
            if line[0] == 'n':
                pass
                
            elif line[0] != '@' :  #start of actual data
                self.featureVectors.append(line.strip().lower().split(','))
            
            else:   #feature definitions
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
                     
        file.close()

    def testingMethod(self, txtfile):       
        file = open(txtfile, 'r')
        for line in file: 
            if line[0] != '@':
                vector = line.strip().lower().split(',')
                print("classifier: %s ," % self.predict(vector) + "given : %s \n" %vector[-1])

    def predict(self, featureVector):               
        probabilityPerLabel = {} 
        for label in self.labelCounts:
            tempProb = 1
            for featureValue in featureVector[:-1]:
                tempProb *= self.featuresDict[label][self.featureNameList[featureVector.index(featureValue)]][featureValue]
            tempProb *= self.labelCounts[label]
            probabilityPerLabel[label]=tempProb
        #print (probabilityPerLabel)
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
        for x in range(0,self.feature):
            self.trainX = self.dictGrad[self.key[x]][:, 0]
            self.trainY = self.dictGrad[self.key[x]][:, 1]
            self.regress[self.key[x]]={}

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
        self.regress[self.key[x]]['Gradien']=self.m

        #calculate coefficient
        x_mean = sum(self.trainX)/n
        y_mean = sum(self.trainY)/n
        self.b=y_mean-(self.m * x_mean)
        self.regress[self.key[x]]['Koefisien']=self.b
        
 
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
        self.regress[self.key[j]]['Gradien']=self.m
        self.regress[self.key[j]]['Koefisien']=self.b
        
    def getValues(self, trainingFile): 
        result = []
        condition = True
        self.regress['title']=trainingFile
        with open(trainingFile,'r') as inputfile:
            for line in inputfile:
                if line[0] == 'L':
                    self.regress['Algoritma']=line
                elif condition == True:
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
    
    def testingMethod(self, trainingFile):
        results1 = []
        with open(trainingFile) as inputfile:
            for line in inputfile:
                if line[0] != '@' and line.split()[0] != 'linear-regression':
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
    def __init__(self,validationFile,model,size=0.7):
        self.trainFile="train.txt"
        self.testFile="test.txt"
        self.file=validationFile
        self.temp=model
        self.size=size
        self.splittingData()
        
    def splittingData(self):        
        with open('data4.txt', "r") as f:
            data = f.read().split()
        select=data[3:]
        title=data[0]
        data.pop(0)
        data.insert(0,title)
        rn.shuffle(select)
        for i in range(0,len(select)):
            data[i+3]=select[i]
        amount=int(len(data)*0.6) #the amount of training data
        train_data = data[:amount]
        test_data = data[amount:]
        
        with open(self.trainFile,"w") as file:
            for i in range (0,len(train_data)-1):
                file.writelines(str(train_data[i]).replace("'","")+"\n")
            file.writelines(str(train_data[-1]).replace("'",""))
            
        with open(self.testFile,"w") as file:
            for i in range (0,len(test_data)-1):
                file.writelines(str(test_data[i]).replace("'","")+"\n")
            file.writelines(str(test_data[-1]).replace("'",""))
         
    def runValidation(self,out):
        self.temp.getValues(self.trainFile)
        self.temp.trainingMethod(out)
        self.temp.testingMethod(self.testFile)
  
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
