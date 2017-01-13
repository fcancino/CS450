import random
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

#Joins the iris data and the iris target
instance_and_target = np.hstack((iris.data, np.atleast_2d(iris.target).T))

#print(instance_and_target)
#shuffles instance_and_target
random.shuffle(instance_and_target)
print(instance_and_target)
#split data into training set and data set
training_set = instance_and_target[0:106, 0:5]
testing_set = instance_and_target[106:151, 0:5]

#Trains data and predicts data
class HardCoded:
    def train(self, training_data_set):
        pass
    def predict(self, data_instances):
        return np.zeros(len(data_instances))


ML = HardCoded()
ML.train(training_set)
array_of_predictions = ML.predict(testing_set)
#I should calculate the percentage of the actual

sum = 0
for i in range(len(testing_set)):
    if testing_set[i][4] == array_of_predictions[i]:
        sum += 1

accuracy_percentage = (sum*100)/len(array_of_predictions)

print(accuracy_percentage)



