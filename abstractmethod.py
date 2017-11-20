from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod
import collections

class MachineLearn(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def TrainingMethod(self):
        pass
    
    @abstractmethod
    def TestingMethod(self):
        pass
    
    @abstractmethod
    def Predict(self):
        pass

class NaiveB(MachineLearn):
    def __init__(self, txtfile):      
        self.trainingFile = txtfile  
        self.featureVectors=[]
        self.features={}
        self.featureNameList=[]
        self.featureCounts = collections.defaultdict(lambda: 1)
        self.labelCounts = collections.defaultdict(lambda: 0)
                 
    def TrainingMethod(self):       
        for fv in self.featureVectors:
            self.labelCounts[fv[len(fv)-1]] += 1 #udpate count of the label
            for counter in range(0, len(fv)-1):
                self.featureCounts[(fv[len(fv)-1], self.featureNameList[counter], fv[counter])] += 1
               
    def GetValues(self):       
        file = open(self.trainingFile, 'r')
        for line in file:
            if line[0] != '@':  #start of actual data
                self.featureVectors.append(line.strip().lower().split(','))
            else:   #feature definitions
                for x in range (0, len(line.strip().split(','))):
                    self.featureNameList.append(line.strip().split(',')[x].replace('@',''))
        z=np.array(self.featureVectors)
        for i in range (0,len(self.featureNameList)):
            y=z[:,i]
            unique=[]
            used = set()
            unique = [x for x in y if x not in used and (used.add(x) or True)]
            self.features[self.featureNameList[i]]=unique  
        file.close()

    def testingMethod(self, txtfile):       
        file = open(txtfile, 'r')
        for line in file: 
            if line[0] != '@':
                vector = line.strip().lower().split(',')
                print ("classifier: " + self.Predict(vector))
        
    def predict(self, featureVector):               
        probabilityPerLabel = {} 
        for label in self.labelCounts:
            tempProb = 1
            for featureValue in featureVector[:-1]:
                tempProb *= self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]/(self.labelCounts[label]+len(self.features[self.featureNameList[featureVector.index(featureValue)]]))
            tempProb *= self.labelCounts[label]
            probabilityPerLabel[label]=tempProb
        #print (probabilityPerLabel)
        return max(probabilityPerLabel, key = lambda classLabel: probabilityPerLabel[classLabel])
          
class Regression(MachineLearn):
    def __init__(self, txtfile):
        self.trainingfile=txtfile
        self.trainX= []
        self.testX= []
        self.trainY= []
        self.testY= []
        self.m=0
        self.b=0
        
    def TrainingMethod(self):
        n=len(self.trainX)
        #calculate gradient
        sum_x= sum(self.trainX)
        sum_y=sum(self.trainY)
        sum_xx=sum(self.trainX* self.trainX)
        sum_xy=sum(self.trainX* self.trainY)
        sxy = sum_xy - (sum_x* sum_y / n)
        sxx = sum_xx - (sum_x* sum_x / n)
        self.m= sxy/sxx
        #calculate coefficient
        x_mean = sum(self.trainX)/n
        y_mean = sum(self.trainY)/n
        self.b=y_mean-(self.m*x_mean)
        print(self.m,':',self.b)
        return [self.m, self.b]
    
    def GetValues(self):
        results1 = []
        with open('data4.txt') as inputfile:
            for line in inputfile:
                results1.append(line.strip().split(','))
        T1 = [list(map(float, x)) for x in results1]
        dataset1= np.array(T1)
        df = pd.DataFrame(dataset1)
        msk = np.random.rand(len(dataset1)) < 0.7
        train = np.array(df[msk])
        test = np.array(df[~msk])
        self.trainX= train[:, 0]
        self.testX= test[:, 0]
        self.trainY=train[:,1]
        self.testY=test[:,1] 
    
    def TestingMethod(self):
        error =0
        for i in range(0,len(self.testY)):
            error += abs(self.testY[i]-self.Predict(i))
        print('Error :',error)

    def Predict(self,i):
        return self.m*self.testX[i]+self.b

if __name__ == "__main__":          
    model = NaiveB("trainset.txt")
    model.GetValues()
    model.TrainingMethod()
    model.TestingMethod("datates.txt")

    Hasil=Regression('data4.txt')
    Hasil.GetValues()
    Hasil.TrainingMethod()
    Hasil.TestingMethod()


