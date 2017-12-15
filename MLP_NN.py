"""
SECTION 1 : Load and setup data for training
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
datatrain = pd.read_csv('irisdataset.csv')

# Change string value to numeric
datatrain.set_value(datatrain['species']=='Iris-setosa',['species'],0)
datatrain.set_value(datatrain['species']=='Iris-versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='Iris-virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)

# Change dataframe to array
datatrain_array = datatrain.as_matrix()

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:4],
                                                    datatrain_array[:,4],
                                                    test_size=0.2)

# Build and Train step
mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print (mlp.score(X_test,y_test))

sl = 6.0
sw = 3.0
pl = 4.8
pw = 1.8
data = np.array([(sl,sw,pl,pw)])
print (mlp.predict(data))

