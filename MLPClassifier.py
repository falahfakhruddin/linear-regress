"""
SECTION 1 : Load and setup data for training
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from numpy.random import shuffle
     
def train_test_split(dataset, test_size = 0.60):
      train_size = int(test_size * len(dataset))
      shuffle(dataset)
      train = dataset[:train_size]
      dataset = np.delete(dataset, np.s_[0:train_size], axis = 0)
      train = np.array_split(train, [-1], axis = 1)
      test = np.array_split(dataset, [-1], axis = 1)
      X_train = np.array(train[0])
      y_train = train[1]
      X_test = test[0]
      y_test = test[1]
      return X_train, X_test, y_train, y_test
      
# Load dataset
datatrain = pd.read_csv('irisdataset.csv')

# Change string value to numeric
datatrain.set_value(datatrain['species']=='Iris-setosa',['species'],0)
datatrain.set_value(datatrain['species']=='Iris-versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='Iris-virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)

# Change dataframe to array
dataset = datatrain.as_matrix()

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(dataset, 
                                                    test_size=0.2)
normalize = Normalization()
normalize.fit(X_train)
X_train = normalize.transform(X_train)
X_test = normalize.transform(X_test)
# Build and Train step
mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=5000)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print (mlp.score(X_test,y_test))

sl = 0.371
sw = 1
pl = 0.085
pw = 0.125
data = np.array([(sl,sw,pl,pw)])
print (mlp.predict(data))

