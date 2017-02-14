import csv
from collections import Counter
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
import math


class Neuron:
    def __init__(self, numOfInputs):
        self.weights = 2 * np.random.random_sample(numOfInputs + 1) - 1
        self.prediction = None
class NeuralNetwork:

    def __init__(self, number_of_layers, nodes_in_layer):
        self.array_of_predictions = []
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
        #array_of_1 = np.empty(len(self.data_train))
        #array_of_1.fill(-1)
        #self.data_train = np.hstack((self.data_train, np.atleast_2d(array_of_1).T))
        #print(self.data_train)
        dict_counter_target = Counter(self.target_train)
        target_counter = 0
        for i in dict_counter_target.keys():
            target_counter += 1

        self.num_of_targets = target_counter
        self.layers = []
        self.nodes_in_layer = nodes_in_layer

        for i in range(1, number_of_layers):
            if i == 1:
                self.layers.append([Neuron(len(self.data_train[0])) for i in range(self.nodes_in_layer)])
            else:
                self.layers.append([Neuron(self.nodes_in_layer) for i in range(self.nodes_in_layer)])

        self.layers.append([Neuron(self.nodes_in_layer) for i in range(self.num_of_targets)])



    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self):
        # for i in range(len(self.neuron_layer1)):
        #     value = np.dot(self.data_train[0], self.neuron_layer1[i].weights)
        #     self.neuron_layer1[i].prediction = self.calculate_sigma(value)

        #train the first layer
            for n in self.data_train:
                for i in self.layers[0]:
                    value = np.dot(np.append(n, - 1), i.weights)
                    i.prediction = self.sigmoid(value)


                for i in range(1, len(self.layers)):
                    predictions = []
                    for n in self.layers[i-1]:
                        predictions.append(n.prediction)

                    predictions = np.array(predictions)


                    for n in self.layers[i]:
                        value = np.dot(np.append(predictions, -1), n.weights)
                        n.prediction = self.sigmoid(value)
                        print(n.prediction)

                predicted_values = []
                for i in self.layers[len(self.layers) - 1]:
                    predicted_values.append(i.prediction)

                predicted_values = np.array(predicted_values)
                prediction = predicted_values.argmax()

                self.array_of_predictions.append(prediction)



        # list_of_predictions = []
        # for i in range(len(self.neuron_layer1)):
        #     list_of_predictions.append(i)
        #
        # print(list_of_predictions)


    def predict(self):
        for prediction in self.array_of_predictions:
            if prediction == 0:
                print("Setosa")
            elif prediction == 1:
                print("Virginica")
            else:
                print("Rubinica")
    def calc_percentage(self):

        sum = 0
        print(self.array_of_predictions)
        for i in range(len(self.target_train)):
            if self.target_train[i] == str(self.array_of_predictions[i]):
                sum += 1
        print(self.target_train)
        return (sum * 100)/ len(self.array_of_predictions)



# for i, value in enumerate(["milk", "bread", "cheese"]):
#   exec ("var%s=value" % (i))


neuralNetwork = NeuralNetwork(3, 4)
neuralNetwork.train()
neuralNetwork.predict()
print(neuralNetwork.calc_percentage())

