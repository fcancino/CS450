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
        pass
    def predict(self, data_test):
        return np.zeros(len(data_test))

#Trains and tests a data set and returns the percentage of accuracy
def calc_accuracy():
    classifier = Classifier()
    classifier.train(data_train, target_train)
    array_of_predictions = classifier.predict(data_test)
    sum = 0
    for i in range(len(array_of_predictions)):
        if array_of_predictions[i] == target_test[i]:
            sum += 1

    return round((sum * 100) / len(array_of_predictions), 2)

print("Accuracy percentage: ", calc_accuracy())


'''User interface'''
class UserInterface():

    algorithm = ""
    data = ""
    split_data = ""

    def main_menu(self):
        print("Machine learning classifier\n")
        print("Options")
        print("Press 'A' to choose your algorithm")
        print("Press 'D' pick the data set to work with")
        print("Press 'S' to choose how to split the data")
        print("Press 'R' to run the mine the data")
        print("Press 'H' to see the menu again")
        print("Press 'P' what you have chosen")
        print("Press 'Q' to quit")

    def user_interaction(self):

        option = ''
        while(option != 'Q' or option != 'q'):
            option = input()
            if(option == 'A' or option == 'a'):
                self.pick_algorithm()
            elif (option == 'D' or option == 'r'):
                self.pick_data()
            elif (option == 'S' or option == 's'):
                self.pick_datasplit()
            elif (option == 'R' or option == 'r'):
                pass
            elif (option == 'H' or option == 'H'):
                self.main_menu()
            elif (option == 'P' or option == 'p'):
                self.print_data()
            elif (option == 'Q' or option == 'q'):
                pass
            else:
                print("Incorrect character")

    def print_data(self):
        print("Algorithm: ", self.algorithm)
        print("Data chose: ", self.data)
        print("Data split method: ", self.split_data)
        self.main_menu()

    def pick_algorithm(self):
        print("Press N for neural network")
        print("Press K for k-Nearest Neighbors")
        print("Press T for Desicion Trees")
        print("Press B for Naive Bayes")
        print("Press Q to go back")
        option = ''
        option = input()
        if (option == 'N' or option == 'n'):
            self.algorithm = "Neural Network"
        elif (option == 'K' or option == 'k'):
            self.algorithm = "K-Nearest Neighbor"
        elif (option == 'T' or option == 't'):
            self.algorithm = "Desicion Trees"
        elif (option == 'B' or option == 'b'):
            self.algorithm = "Naive Bayes"
        elif (option == 'Q' or option == 'q'):
            pass
        else:
            print("Incorrect character\n")

        self.main_menu()

    def pick_data(self):
        self.data = input("Type type of data input: ")
        self.main_menu()

    def pick_datasplit(self):
        self.split_data = input("Type how do you want to split the data: ")
        self.main_menu()

#Uncomment to run
#userInterface = UserInterface()
#userInterface.main_menu()
#userInterface.user_interaction()

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