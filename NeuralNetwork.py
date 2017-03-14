import csv
from collections import Counter
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
import math

LEARNING_RATE = 0.3

class Neuron:
    def __init__(self, numOfInputs):
        self.weights = 2 * np.random.random_sample(numOfInputs + 1) - 1
        self.prediction = None
        self.error = None
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
        # I'll call both in a loop 10,000 times
        for i in range(400):
            for k in range(len(self.data_train)):
                self.forward_phase(self.data_train[k])
                self.backwards_phase(self.target_train[k], self.data_train[k])

            if (i % 10) == 0:
                self.predict()
                print("Accuracy of the neural network:")
                print(self.calc_percentage())



    def forward_phase(self, data_item, test_prediction=False):
        # for i in range(len(self.neuron_layer1)):
        #     value = np.dot(self.data_train[0], self.neuron_layer1[i].weights)
        #     self.neuron_layer1[i].prediction = self.calculate_sigma(value)

        #train the first layer
        for i in self.layers[0]:

            value = np.dot(np.insert(data_item, 0, - 1.0), i.weights)
            i.prediction = self.sigmoid(value)


        for i in range(1, len(self.layers)):
            #get the prediction of the previous layer
            predictions = []
            for n in self.layers[i-1]:
                predictions.append(n.prediction)
            #put the prediction in a numpy array
            predictions = np.array(predictions)

            #train the current layer
            for n in self.layers[i]:
                #dot product
                value = np.dot(np.insert(predictions,0, -1), n.weights)

                n.prediction = self.sigmoid(value)

        #predicted values
        predicted_values = []
        # for i in self.layers[len(self.layers) - 1]:
        #     predicted_values.append(i.prediction)
        # predicted_values = np.array(predicted_values)
        # prediction = predicted_values.argmax()
        #
        # self.array_of_predictions.append(prediction)

        # for i in range(len(self.layers)):
        #     for j in self.layers[i]:
        #         print(j.weights)
        if test_prediction == True:
            list_of_predictions = []
            for i in range(len(self.neuron_layer1)):
                list_of_predictions.append(i)

    def backwards_phase(self, training_target_item, training_data_item):
        # creates a list based on the targets

        output_target = [0]*self.num_of_targets
        for i in range(len(output_target)):
            if training_target_item == str(i):
                output_target[i] = 1


        #output layer error
        for i in range(len(self.layers[len(self.layers)-1])):

            #calculate error in each of the output nodes
            #prediction for debugging purposes
            prediction = self.layers[len(self.layers) - 1][i].prediction
            #error of the node is: activation * (1 - activation) * (activation - target)
            self.layers[len(self.layers)-1][i].error = (self.layers[len(self.layers)-1][i].prediction
                                                        - output_target[i])*self.layers[len(self.layers)-1][i].prediction*(1 -
                                                       self.layers[len(self.layers) - 1][i].prediction)


        #calculate error in the hidden layers
        #error = node_output*(1-node_output)*sum(node_weights*right_errors)
        for i in reversed(range(len(self.layers))):
            if i - 1 >= 0:
                for n in range(len(self.layers[i-1])):
                    error_right_layer = 0
                    for k in range(len(self.layers[i])):
                            error_right_layer += self.layers[i][k].error * self.layers[i][k].weights[n+1]
                    prediction2 = self.layers[i - 1][n].prediction
                    self.layers[i - 1][n].error = self.layers[i-1][n].prediction * (1 - self.layers[i-1][n].prediction) * error_right_layer

        #update weights

        #print(len(self.layers[1][0].weights))
        #change in weights in layer:

        for i in reversed(range(len(self.layers))):
            for k in range(len(self.layers[i])):
                error = self.layers[i][k].error
                for n in range(len(self.layers[i][k].weights)):
                    output_left_node = None
                    if n == 0:
                        # for first weight, always combined with the bias node
                        output_left_node = -1
                    elif i == 0 and n != 0:
                        #first layer
                        output_left_node = training_data_item[n-1]
                    else:
                        #rest of the layers
                        output_left_node = self.layers[i - 1][n-1].prediction
                    weight = self.layers[i][k].weights[n]
                    weight = weight - LEARNING_RATE * output_left_node * error
                    self.layers[i][k].weights[n] = weight


        #debugging purposes:

    def predict(self):
        # predicted values
        self.array_of_predictions = []
        for i in range(len(self.data_test)):
            self.forward_phase(self.data_test[i], test_prediction=False)
            predicted_values = []
            for i in self.layers[len(self.layers) - 1]:
                predicted_values.append(i.prediction)


            predicted_values = np.array(predicted_values)
            prediction = predicted_values.argmax()

            self.array_of_predictions.append(prediction)

    def calc_percentage(self):

        sum = 0

        for i in range(len(self.target_test)):
            if self.target_test[i] == str(self.array_of_predictions[i]):
                sum += 1
        return (sum * 100)/ len(self.array_of_predictions)





neuralNetwork = NeuralNetwork(2, 4)
#neuralNetwork.forward_phase([0.4, 0.3, 0.2, 0.3])
neuralNetwork.train()
# neuralNetwork.train()



