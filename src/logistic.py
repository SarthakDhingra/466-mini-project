#! /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# alpha values to consider
alphas = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3]      # learning rate

# function to select best hyperparameters
def optimize_logistic_regression(data):

    # intialize things
    w_best = None
    b_best = None
    best_alpha = None
    best_accuracy = float('-inf')
    best_losses = None

    # try all learning rate alphas
    for alpha in alphas:

        # train and validate
        w, b, losses,  = train_logistic_regression(data, alpha)
        t_test_hat, _ = predict_logistic_regression(data['X_validation'], w, b)
        current_accuracy = get_accuracy(t_test_hat, data['y_validation'])

        # update best values
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            w_best = w 
            b_best = b
            best_alpha = alpha
            best_losses = losses
    plot_loss(best_losses)
    return w_best, b_best, alpha, best_accuracy

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
    losses_train = []

    # iterate through epochs
    for i in range(epoch):
        
        # mini-batch gradient descent
        loss_this_epoch = 0
        for batch in range( int(np.ceil(N_train/batch_size)) ):
            
            # get batch
            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size : (batch+1)*batch_size]

            # get prediction 
            y_hat, y_hat_original = predict_logistic_regression(X_batch, w, b)
            
            # get loss
            batch_loss = 0
            for i in range(len(y_hat_original)):
                batch_loss += cross_entropy(t_batch[i], y_hat_original[i])
            loss_this_epoch += batch_loss

            # calculate sigmoid
            z = np.dot(X_batch, w) + b
            y = 1 / (1 + np.exp(-z))

            # calculate gradients
            w_grad = np.dot(X_batch.T, (y-t_batch))
            b_grad = np.sum(y-t_batch)
            
            # update weights and bias
            w = w - alpha*w_grad
            b = b - alpha*b_grad
        
        # append loss
        training_loss = loss_this_epoch / int(np.ceil(N_train/batch_size))
        losses_train.append(training_loss)

        # keep track of best accuracy, and update best weights and bias if need be
        t_val_hat, _ = predict_logistic_regression(data['X_validation'], w, b)
        current_accuracy = get_accuracy(t_val_hat, data['y_validation'])
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            w_best = w 
            b_best = b
    
    return w_best, b_best, losses_train

# function to compute logistic regression
def predict_logistic_regression(X, w, b):

    # linear regression
    z = np.dot(X,w) + b
    # sigmoid
    t = 1 / (1 + np.exp(-z))
    # binarize values
    t_final = vbinarize(t)

    return t_final, t

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

# function to get cross entropy loss of one sample
def cross_entropy(t, t_hat):
    if t == 1:
        return -np.log(t_hat)
    else:
        return -np.log(1 - t_hat)

vbinarize = np.vectorize(binarize)

# function to plot losses vs. epoch
def plot_loss(losses):

    # create epochs array
    epochs = range(len(losses))

    # create plot
    plt.figure()
    plt.scatter(epochs, losses, s=10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.title(f'Logistic Regression Epochs vs. Loss')
    plt.savefig(f'logistic_loss.png')