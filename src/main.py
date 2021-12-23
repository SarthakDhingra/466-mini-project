#! /usr/bin/env python3

# supress pandas future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import libraries
import numpy as np
import pandas as pd

# import models
from logistic import predict_logistic_regression, optimize_logistic_regression, get_accuracy
from knn import train_knn
from nn import train_nn

def load_data():

    # read CSV
    DF = pd.read_csv('../data/diabetes.csv')

    # replace bad 0 values with NaN
    columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    DF[columns] = DF[columns].replace({0:np.NaN}) 

    # replace NaN values with column means    
    for col in columns:
        DF[col].fillna((DF[col].mean()), inplace=True)
    
    # get input and output
    X = np.asarray(DF.drop('Outcome', 1))
    y = np.asarray(DF['Outcome'])

    # normalize data
    mean_x  = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    X = (X - mean_x) / std_x

    # randomly shuffle data
    np.random.seed(314)
    np.random.shuffle(X)
    np.random.seed(314)
    np.random.shuffle(y)

    # 60% train, 20% validation, 20% test
    X_train = X[:460]
    y_train = y[:460]

    X_validation = X[460:613]
    y_validation = y[460:613]

    X_test = X[613:]
    y_test = y[613:]
    
    data = {'X_train':X_train, 'y_train':y_train, 'X_validation':X_validation, 'y_validation':y_validation, 'X_test':X_test, 'y_test':y_test}
    return data 

def driver(data):

    # # neural network
    # network, num_nodes, activation = train_nn(data)
    # _, accuracy = network.evaluate(data['X_test'], data['y_test'])
    # print(f"Accuracy of neural network is {accuracy} using {num_nodes} nodes in the hidden layer and {activation} activation")

    # # logistic regression
    # w, b, alpha = optimize_logistic_regression(data)
    # t_hat = predict_logistic_regression(data['X_test'], w, b)
    # print(f"Accuracy of logistic regression is {get_accuracy(t_hat, data['y_test'])} with best alpha {alpha}")
    
    # knn
    knn, best_neighbour, best_distance, best_validation = train_knn(data)
    t_hat = knn.predict(data['X_test'])
    print(f"Accuracy of knn is {get_accuracy(t_hat, data['y_test'])} with {best_neighbour} neighbours using {best_distance} distance. Validation accuracy was {best_validation}")

    # majority guess
    y_test = data['y_test']
    majority_guess = (y_test == 0).sum() / len(y_test)
    print(f"Accuracy of majority guess: {majority_guess}")
   

if __name__ == '__main__':
    data = load_data()
    driver(data)