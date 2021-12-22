#! /usr/bin/env python3
import numpy as np
import pandas as pd
# function to binarize data to 0 or 1
def binarize(x):
    if x<0.5:
        return 0
    else:
        return 1

vbinarize = np.vectorize(binarize)

# function to train logistic regression
def train_logistic_regression(data):

    X = data['X_train']
    t = data['y_train']

    # initialize things
    epoch = 100
    alpha = 0.001 # learning rate
    b = 0
    w = np.ones([X.shape[1]])
    w_best = w 
    b_best = b
    best_accuracy = 0

    for i in range(epoch):
        # predict 
        y_hat = predict_logistic_regression(X, w, b)

        # y is the sigmoid
        z = np.dot(X, w) + b
        y = 1 / (1 + np.exp(-z))

        # calculate gradients
        w_grad = np.dot(X.T, (y-t))
        b_grad = np.sum(y-t)
        
        # updatew weights and bias
        w = w - alpha*w_grad
        b = b - alpha*b_grad

        # keep track of best accuracy, and update best weights and bias if need be
        current_accuracy = get_accuracy(y_hat, t)
        print(current_accuracy)
        # print(f'current_accuracy = {current_accuracy}')
        if current_accuracy > best_accuracy:
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