#! /usr/bin/env python3
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

distances = ["euclidean", "manhattan", 'chebyshev']

def train_knn(data):

    # get training and validation
    X_train = data['X_train']
    y_train = data['y_train']
    X_validation = data['X_validation']
    y_validation = data['y_validation']

    # best parameters
    best_neighbour = None
    best_distance = None
    best_model = None
    best_accuracy = float('-inf')

    # for plotting accuracies
    euclidean_accuracies = []
    manhattan_accuracies = []
    chebyshev_accuracies = []

    for distance in distances:
        for i in range(1,101):

            # training
            knn = KNeighborsClassifier(n_neighbors=i, metric=distance)
            knn.fit(X_train, y_train)

            # validation
            y_val_pred = knn.predict(X_validation)
            accuracy = get_accuracy(y_val_pred, y_validation)

            # append accuracies
            if distance == "euclidean":
                euclidean_accuracies.append(accuracy)
            elif distance == "manhattan":
                manhattan_accuracies.append(accuracy)
            else:
                chebyshev_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_distance = distance
                best_neighbour = i
                best_model = knn
    
    plot_accuracies(euclidean_accuracies, "Euclidean")
    plot_accuracies(manhattan_accuracies, "Manhattan")
    plot_accuracies(chebyshev_accuracies, "Chebyshev")
    return best_model, best_neighbour, best_distance, best_accuracy

# get accuracy
def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    correct = np.sum(t == t_hat)
    total = len(t)
    acc = correct / total

    return acc

# function to plot accuracies
def plot_accuracies(accuracies, name):

    # get k array
    ks = []
    for i in range(len(accuracies)):
        ks.append(i+1)

    # create plot
    plt.figure()
    plt.scatter(ks, accuracies, s = 20)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title(f'{name} Distance Accuracies vs. K')
    plt.savefig(f'knn_{name}.png')



    


