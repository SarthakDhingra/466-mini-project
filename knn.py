#! /usr/bin/env python3
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

distances = ["euclidean", "manhattan", "minkowski"]

def train_knn(data):

    X_train = data['X_train']
    y_train = data['y_train']
    X_validation = data['X_validation']
    y_validation = data['y_validation']

    best_neighbour = None
    best_distance = None
    best_model = None
    best_accuracy = float('-inf')

    for distance in distances:
        for i in range(1,101):

            # training
            knn = KNeighborsClassifier(n_neighbors=i, metric=distance)
            knn.fit(X_train, y_train)

            # validation
            y_val_pred = knn.predict(X_validation)
            accuracy = get_accuracy(y_val_pred, y_validation)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_distance = distance
                best_neighbour = i
                best_model = knn
    
    return best_model, best_neighbour, best_distance

# should move to utils file
def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    correct = np.sum(t == t_hat)
    total = len(t)
    acc = correct / total

    return acc


    


