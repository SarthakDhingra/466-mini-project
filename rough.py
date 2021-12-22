#! /usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

debug = True

def rough():
    X, y = load_diabetes(return_X_y=True)
    print('sklearn info')
    print(X[0])

    # load raw CSV from pandas
    # raw csv was used since sklearn gives already-normalized data
    # add link to where data comes from
    df = pd.read_csv('diabetes.tab', delimiter='\t')
    y = df.Y.values
    X = df.drop(['Y'], axis=1)

    # next normalize data
    # mean_x  = np.mean(X, axis=0)
    # std_x = np.std(X, axis=0)
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # can remove random_state for end
    # createsa an 80% train, 10% validation, 10% test split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation, y_validation, test_size=0.5, random_state=1)

    # print(X_train.shape)
    # print(y_train.shape)

    # print(X_validation.shape)
    # print(y_validation.shape)

    # print(X_test.shape)
    # print(y_test.shape)
def load_pima():
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
    
    w, b = train_logistic_regression(X_train, y_train, X_validation, y_validation)
    t_hat = predict_logistic_regression(X_test, w, b)
    print("Accuracy of logistic regression on test set", get_accuracy(t_hat, y_test))



def binarize(x):
    if x<0.5:
        return 0
    else:
        return 1

vbinarize = np.vectorize(binarize)

def train_logistic_regression(X_train, t_train, X_validation, t_validation):
    N_train = X_train.shape[0]
    # initialize things
    epoch = 100
    alpha = 0.001 # learning rate
    b = 0
    w = np.ones([X_train.shape[1]])
    w_best = w 
    b_best = b
    best_accuracy = 0
    batch_size   = 10    # batch size

    for i in range(epoch):
        # predict 
        for batch in range( int(np.ceil(N_train/batch_size))):

            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size : (batch+1)*batch_size]

            y_hat = predict_logistic_regression(X_batch, w, b)

            # y is the sigmoid
            z = np.dot(X_batch, w) + b
            y = 1 / (1 + np.exp(-z))

            # calculate gradients
            w_grad = np.dot(X_batch.T, (y-t_batch))
            b_grad = np.sum(y-t_batch)
            
            # updatew weights and bias
            w = w - alpha*w_grad
            b = b - alpha*b_grad
        
        # keep track of best accuracy, and update best weights and bias if need be
        y_val_hat = predict_logistic_regression(X_validation, w, b)
        current_accuracy = get_accuracy(y_val_hat, t_validation)
        # print(f'current_accuracy = {current_accuracy}')
        if current_accuracy < best_accuracy:
            best_accuracy = current_accuracy
            w_best = w 
            b_best = b
    
    return w_best, b_best


def predict_logistic_regression(X, w, b):

    z = np.dot(X,w) + b
    t = 1 / (1 + np.exp(-z))
    t = vbinarize(t)

    return t

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    # print(t.shape)
    # print(t_hat.shape)
    # confirm
    # t and t_hat should be same size
    correct = np.sum(t == t_hat)
    total = len(t)
    acc = correct / total
    return acc


if __name__ == '__main__':
    #rough()
    load_pima()