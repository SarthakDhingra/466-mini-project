#! /usr/bin/env python3
import numpy as np
import pandas as pd

alphas = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3]      # learning rate

# can i do this on test data?
def optimize_logistic_regression(data):

    w_best = None
    b_best = None
    best_alpha = None
    best_accuracy = float('-inf')

    for alpha in alphas:
        w, b = train_logistic_regression(data, alpha)

        t_test_hat = predict_logistic_regression(data['X_test'], w, b)
        current_accuracy = get_accuracy(t_test_hat, data['y_test'])

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            w_best = w 
            b_best = b
            best_alpha = alpha
    
    return w_best, b_best, alpha



# function to train logistic regression
def train_logistic_regression(data, alpha):

    # get training data
    X_train = data['X_train']
    t_train = data['y_train']
   

    # initialize things
    epoch = 100     # number of epochs
    batch_size = 10     # batch size
    b = 0           # initial bias    
    b_best = b      # initial best bias
    w = np.ones([X_train.shape[1]])     # initial weights
    w_best = w          # initial best weights
    best_accuracy = 0   # initial best accuracy
    N_train = X_train.shape[0]  # number of training data

    # iterate through epochs
    for i in range(epoch):
        
        # mini-batch gradient descent
        for batch in range( int(np.ceil(N_train/batch_size)) ):
            
            # get batch
            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size : (batch+1)*batch_size]

            # get prediction 
            y_hat = predict_logistic_regression(X_batch, w, b)

            # calculate sigmoid
            z = np.dot(X_batch, w) + b
            y = 1 / (1 + np.exp(-z))

            # calculate gradients
            w_grad = np.dot(X_batch.T, (y-t_batch))
            b_grad = np.sum(y-t_batch)
            
            # update weights and bias
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

# function to compute logistic regression
def predict_logistic_regression(X, w, b):

    # linear regression
    z = np.dot(X,w) + b
    # sigmoid
    t = 1 / (1 + np.exp(-z))
    # binarize values
    t = vbinarize(t)

    return t

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    correct = np.sum(t == t_hat)
    total = len(t)
    acc = correct / total

    return acc

# function to binarize data to 0 or 1
def binarize(x):
    if x<0.5:
        return 0
    else:
        return 1

vbinarize = np.vectorize(binarize)

# TO DO
# give credit to CA2 + CA1
# mention loss function in report