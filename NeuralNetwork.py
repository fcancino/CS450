import csv
from collections import Counter
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random

class Neuron:
    def __init__(self, numOfInputs):
        self.weights = 2 * np.random.random_sample(numOfInputs) - 1
        self.prediction = None
class NeuralNetwork:

    def __init__(self):

        temp_data = []
        temp = []
        with open('iris.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile)

            for row in spamreader:
                temp_data.append(row)

        features = np.array(temp_data)

        data_instances = features[:,:-1]
        len_dataset_row = len(features[0])
        targets = features[:,len_dataset_row - 1]

        count_data_items = Counter(targets)
        data_targets = []
        for i in count_data_items.keys():
            data_targets.append(i)

        for i in range(len(targets)):
            for n in range(len(data_targets)):
                if (targets[i] == data_targets[n]):
                    targets[i] = n

        normalized_data_instances = preprocessing.normalize(data_instances)

        self.data_train, self.data_test, self.target_train, self.target_test = tts(normalized_data_instances, targets, train_size=.7)
        array_of_1 = np.empty(len(self.data_train))
        array_of_1.fill(-1)
        self.data_train = np.hstack((self.data_train, np.atleast_2d(array_of_1).T))
        print(self.data_train)
        dict_counter_target = Counter(self.target_train)
        target_counter = 0
        for i in dict_counter_target.keys():
            target_counter += 1

        self.num_of_targets = target_counter
        self.neuron_layers = [Neuron(len(self.data_train[0])) for i in range(self.num_of_targets)]


    def train(self):
        for i in range(len(self.neuron_layers)):
            value = np.dot(self.data_train[0], self.neuron_layers[i].weights)
            self.neuron_layers[i].prediction = value >= 0

    def predict(self):
        prediction_num = 0
        for i in range(len(self.neuron_layers)):
            if self.neuron_layers[i].prediction:
                prediction_num += 1

        if prediction_num == 0:
            return "Type 0"
        elif prediction_num == 1:
            return "Type 1"
        else:
            return "Other type"




neuralNetwork = NeuralNetwork()

neuralNetwork.train()
print(neuralNetwork.predict())
