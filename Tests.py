import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
'''Classifier'''
#loads the iris data
iris = datasets.load_iris()

#separates and randomizes the data
data_train, data_test, target_train, target_test = tts(iris.data, iris.target, train_size = .7)

#Trains data and predicts data
class Classifier:

    def train(self, data_train, target_train):
        self.data_train = data_train
        self.target_train = target_train
    def predict(self, data_test, num_neighbors):
        data_test_length = len(data_test)
        array_of_distances = np.zeros(len(self.data_train))
        predictions_array = np.zeros(len(self.data_train))
        array_of_indexes = np.zeros(num_neighbors)
        distance = 0
        for i in range(data_test_length):
            #calculates the distance
            for k in range(len(data_train)):
                distance = 0
                for j in range(4):
                    distance += (data_test[i][j] - self.data_train[k][j])**2
                euclidean_distance = distance**1/2
                array_of_distances[k] = euclidean_distance

            #creates an array of the values of the nearest neighbors
            for n in range(num_neighbors):
                min_index = np.argmin(array_of_distances)
                array_of_indexes[n] = min_index
                max_index = np.argmax(array_of_distances)
                array_of_distances[min_index] = array_of_distances[max_index] + 1

            temp_list = []

            for h in array_of_indexes:
                temp_list.append(target_train[h])

            #returns the most common value of the list of the values of the closest neighbors
            mostcommon = max(set(temp_list), key=temp_list.count)
            predictions_array[i] = mostcommon





        return predictions_array


#Trains and tests a data set and returns the percentage of accuracy
def calc_accuracy():
    classifier = Classifier()
    classifier.train(data_train, target_train)
    array_of_predictions = classifier.predict(data_test, 3)
    sum = 0
    for i in range(45):
        if array_of_predictions[i] == target_test[i]:
            sum += 1
    return (sum*100.0)/len(data_test)


print(calc_accuracy())

