import numpy as np
import csv
import math
from collections import Counter
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tts
from random import randint

class Node:

    def __init__(self):
        self.name = ""
        self.column = None
        self.targets = None
        self.child_left = None
        self.child_middle = None
        self.child_middle_right = None
        self.child_right = None
        self.entropy = None
        self.isLeaf = False
        self.data = None
        self.value = None


class DesicionTree:

    def __init__(self):
        self.root = None
        self.node = Node()

        self.readFile()
        self.test_train(self.node)


    def readFile(self):
        temp_data = []
        temp = []
        with open('iris1.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile)

            for row in spamreader:
                temp_data.append(row)

        features = np.array(temp_data)

        data_instances = features[:, [0, 1, 2, 3]]
        targets = features[:,4]

        self.data_train, self.data_test, self.target_train, self.target_test = tts(data_instances, targets, train_size=.7)

    '''Ideas for the node
        1. calculate the entropy of each column. how? Put each column element in a node with the targets(maybe). For every different value add childs
        for every child add different values. Calculate the total entropy and save it.
        (Keep in count that I want to leave some data of the training set to mini test my algorithm)

    '''

    def test_train(self, node, items_not_left=0):

        num_items = 0
        if(node.data != None):
            quantity_data = Counter(node.data)
            for i in quantity_data.values():
                num_items += i
                break

            if (num_items == len(node.data)):
                node.isLeaf = True
                node.value = "setosafornow"
                return

        elif (items_not_left == len(self.data_train[0])):
            node.isLeaf = True
            quantity_data = Counter(node.data)
            list_of_values = []
            for i in quantity_data.values():
                list_of_values.append(i)
            if (len(set(list_of_values)) == 1):
                index = randint(0, len(node.data))
                node.value = "Momentaruis"
                print(node.value)
                return
            #TODO add logic to set the value as the most common.


        else:
            for cycle in range(items_not_left , len(self.data_train[0])):
                node.data = self.data_train[:,cycle]
                most_common_items = Counter(node.data)
                num_diff_items = len(most_common_items.keys())  # I can set the num of childs of the node
                node.data = self.data_train[:, [cycle]].tolist()

                for i in range(len(node.data)):
                    node.data[i].append(i)

                print(node.data)
                values = []
                for key in most_common_items.keys():
                    values.append(key)

            #for i in range(len(column_holder)):
              #  for n in range(le  n(values)):

            for i in range(len(column_holder)):
                if(column_holder[i][0] == values[0]):
                    list_left.append(column_holder[i])
                    print("Left")
                    #break
                if(column_holder[i][0] == values[1]):
                    print("Middle")
                    list_middle.append(column_holder[i])
                    #break
                if(column_holder[i][0] == values[2]):
                    print("Middle-Right")
                    list_middle_right.append(column_holder[i])
                    #break
                if (len(values) == 4):
                    if (column_holder[i][0] == values[3]):
                        print("Right")
                        list_right.append(column_holder[i])


            #May want to set a value for how many items I am calculating the entropy

            count_targets_in_node_left = []
            count_targets_in_node_middle = []
            count_targets_in_node_middle_right = []
            count_targets_in_node_right = []
            for i in range(len(list_left)):
                count_targets_in_node_left.append(self.target_train[list_left[i][1]])
            for i in range(len(list_middle)):
                count_targets_in_node_middle.append(self.target_train[list_middle[i][1]])
            for i in range(len(list_middle_right)):
                count_targets_in_node_middle_right.append(self.target_train[list_middle_right[i][1]])
            if(len(values) == 4):
                for i in range(len(list_right)):
                    count_targets_in_node_right.append(self.target_train[list_right[i][1]])

            #calculate the entropy of each node
            count_left = Counter(count_targets_in_node_left)
            print(count_left)




    def calc_entropy(self, list_targets):
        targets = []
        entropy = 0
        quant_targets_dict = Counter(list_targets)
        num_targets = len(list_targets)

        for i in quant_targets_dict.values():
            targets.append(i)

        for n in targets:
            entropy -= (n / num_targets) * math.log2(n / num_targets)

        print(entropy)

        return entropy

    def train(self):
        '''trains the data using desicion tree
        if node.data is all the same
            return i don't know what
        else if
            return node.data most common label

        else
            choose a feature that maximizes the information gain
            this will be the next node to use

            add a branch to the node  for each possible value in F.

            calculate sf

            call the function again.



        '''



dTree = DesicionTree()
#dTree.readFile()
#dTree.test_train()