import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts

#loads the iris data
iris = datasets.load_iris()

#separates and randomizes the data
data_train, data_test, target_train, target_test = tts(iris.data, iris.target, train_size= .7)

'''
This was my first try in separating and randomizing the data
#Joins the iris data and the iris target
instance_and_target = np.hstack((iris.data, np.atleast_2d(iris.target).T))

#shuffles instance_and_target
random.shuffle(instance_and_target)

#split data into training set and data set
training_set = instance_and_target[0:106, 0:5]
testing_set = instance_and_target[106:151, 0:5]
'''

#Trains data and predicts data
class HardCoded:
    def train(self, data_train, target_train):
        pass
    def predict(self, data_test):
        return np.zeros(len(data_test))


ML = HardCoded()
ML.train(data_train, target_train)
array_of_predictions = ML.predict(data_test)


sum = 0
for i in range(len(array_of_predictions)):
   if array_of_predictions[i] == target_test[i]:
       sum += 1

accuracy_percentage = round((sum*100)/len(array_of_predictions), 2)

print(accuracy_percentage)