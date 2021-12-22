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

    X_train = data['X_train']
    t_train = data['y_train']
    N_train = X_train.shape[0]

    # initialize things
    epoch = 100
    alpha = 0.001 # learning rate
    b = 0
    w = np.ones([X_train.shape[1]])
    w_best = w 
    b_best = b
    best_accuracy = 0
    batch_size = 10

    for i in range(epoch):

        for batch in range( int(np.ceil(N_train/batch_size)) ):

            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size : (batch+1)*batch_size]

            # predict 
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
        t_val_hat = predict_logistic_regression(data['X_validation'], w, b)
        current_accuracy = get_accuracy(t_val_hat, data['y_validation'])
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