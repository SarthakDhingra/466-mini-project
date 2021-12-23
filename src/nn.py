#! /usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt

# different activiation functions to consider
activations = ['relu', 'sigmoid', 'tanh']

# function to train nn
def train_nn(data):

    # get training and validation information
    X_train = data['X_train']
    y_train = data['y_train']
    X_validation = data['X_validation']
    y_validation = data['y_validation']

    # initialize parameters
    best_accuracy = float('-inf')
    best_nodes = None
    best_activation = None
    best_model = None
    best_losses = None

    # iterate through all nodes
    for nodes in range(2, 100, 4):

        # iterate through all activations
        for activation in activations:
            
            # print statement
            print(f'Processing {nodes} nodes with {activation} activation')

            # sequential network
            # groups linear stack of layers
            model = Sequential()

            # middle layer 8 input nodes --> X nodes
            model.add(Dense(nodes, input_dim=8, activation=activation))

            # final layer with 1 node
            model.add(Dense(1, activation='sigmoid'))

            # loss, optimizer, and metric
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # get history of training
            history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_validation, y_validation), verbose=0)
            
            # update accuracy
            _, accuracy = model.evaluate(data['X_validation'], data['y_validation'])

            # update best values
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_nodes = nodes
                best_activation = activation
                best_model = model
                best_losses = history.history['loss']

    # plot loss vs. epoch
    plot_loss(best_losses)
    return best_model, best_nodes, best_activation, best_accuracy

# function to plot losses vs. epoch
def plot_loss(losses):

    # create epochs array
    epochs = range(len(losses))

    # create plot
    plt.figure()
    plt.scatter(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.title(f'Best Neural Network Epochs vs. Loss')
    plt.savefig(f'nn_loss.png')


