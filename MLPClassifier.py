from sklearn.neural_network import MLPClassifier
from Abstraction import AbstractML
from DatabaseConnector import DatabaseConnector


class SklearnNeuralNet(AbstractML):
    def __init__(self,size=10, solver='sgd', learning_rate=0.01, max_iter=5000):
        self.mlp = MLPClassifier(hidden_layer_sizes=size, solver=solver, learning_rate_init=learning_rate, max_iter=max_iter)

    def training(self, features, target):
        self.mlp.fit(features, target)
        return self.mlp

    def predict(self, features, model=None):
        if model != None:
            self.mlp = model
        prediction = (self.mlp.predict(features))
        prediction = prediction.tolist()
        return prediction

    def testing(self, features, target, model=None):
        if model != None:
            self.mlp = model
        error = self.mlp.score(features, target)
        return error

if __name__ == "__main__":
    db = DatabaseConnector()
    df = db.get_collection("irisdataset", "species")
    features = df[0]
    target = df[1]
    header = df[2]

    #training
    nn = SklearnNeuralNet()
    model = nn.training(features, target)
    model2 = nn
    print(model2.predict(features))

    list_model = ["irisdataset", SklearnNeuralNet, nn]

"""
json_attr = json.dumps(attr)


import pickle
filename = "sklearn_model.txt"
binary = pickle.dumps(mlp)


# Test the model
print (mlp.score(X_test,y_test))

sl = 0.371
sw = 1
pl = 0.085
pw = 0.125
data = np.array([(sl,sw,pl,pw)])
prediction = (mlp.predict(data))

from sklearn.preprocessing import LabelBinarizer
label = LabelBinarizer()
y_train = label.fit(y_train)
label.classes_
label.transform(y_test)
len(target[0])
target = target.reshape(len(target[0]),1)
j =0
for i in range (0, len(prediction)):
    if prediction[i] != target[i]:
        j += 1

print (j)



db = DatabaseConnector()
result2 = db.import_collection(dict_binary,'sklearn',database='newdb')
dict_binary = {"data":binary}
client = MongoClient()
db = client['newdb']
upload = db['mlp']
hasil = upload.insert(dict_binary)

db = client['newdb']
collection = db['sklearn'].find_one()
copy_binary = collection['data']

pls = pickle.loads(copy_binary)

pls.predict(features)




#datatrain = pd.read_csv('irisdataset.csv')
client = MongoClient()
db = client['newdb']
collection = db['irisdataset'].find()
datatrain = pd.DataFrame(list(collection))
del datatrain['_id']

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
"""

