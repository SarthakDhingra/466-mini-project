#! /usr/bin/env python3

# supress pandas future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import libraries
import numpy as np
import pandas as pd

from logistic import predict_logistic_regression, optimize_logistic_regression, get_accuracy

# flags
debug = False

#TODO
# replace 0 values with mean?
def load_data():
    DF = pd.read_csv('diabetes.csv')
    X = np.asarray(DF.drop('Outcome', 1))
    y = np.asarray(DF['Outcome'])

    if debug:
        print()
        print(f'DF shape = {DF.shape}')
        print(f'X shape = {X.shape}')
        print(f'y shape = {y.shape}')
        print()


    # normalize data
    mean_x  = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    X = (X - mean_x) / std_x

    # 60% train, 20 validation, 20 test
    # should use sklearn later to randomize
    X_train = X[:460]
    y_train = y[:460]

    X_validation = X[460:613]
    y_validation = y[460:613]

    X_test = X[613:]
    y_test = y[613:]

    if debug:
        print(f'X_train shape = {X_train.shape}')
        print(f'y_train shape = {y_train.shape}')
        print(f'X_validation shape = {X_validation.shape}')
        print(f'y_validation shape = {y_validation.shape}')
        print(f'X_test shape = {X_test.shape}')
        print(f'y_test shape = {y_test.shape}')
        print()
    
    data = {'X_train':X_train, 'y_train':y_train, 'X_validation':X_validation, 'y_validation':y_validation, 'X_test':X_test, 'y_test':y_test}
    return data 

def driver(data):

    # majority guess
    y_test = data['y_test']
    majority_guess = (y_test == 0).sum() / len(y_test)
    print(f"Accuracy of majority guess: {majority_guess}")

    # logistic regression
    w, b, alpha = optimize_logistic_regression(data)
    t_hat = predict_logistic_regression(data['X_test'], w, b)
    print(f"Accuracy of logistic regression is {get_accuracy(t_hat, data['y_test'])} with best alpha {alpha}")
    

if __name__ == '__main__':
    data = load_data()
    driver(data)