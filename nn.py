#! /usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense

activations = ['relu', 'sigmoid', 'tanh']
def train_nn(data):

    X_train = data['X_train']
    y_train = data['y_train']
    X_validation = data['X_validation']
    y_validation = data['y_validation']

    best_accuracy = float('-inf')
    best_nodes = None
    best_activation = None
    best_model = None

    for nodes in range(2, 10, 10):
        for activation in activations:
            print(f'Processing {nodes} nodes with {activation} activation')

            # sequential network
            model = Sequential()

            # middle layer 8 input nodes --> X nodes
            model.add(Dense(nodes, input_dim=8, activation=activation))

            # final layer with 1 node
            model.add(Dense(1, activation='sigmoid'))

            # loss, optimizer, and metric
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_validation, y_validation), verbose=0)
            
            # update accuracy
            _, accuracy = model.evaluate(data['X_validation'], data['y_validation'])

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_nodes = nodes
                best_activation = activation
                best_model = model

    return best_model, best_nodes, best_activation