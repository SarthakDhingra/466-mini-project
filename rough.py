#! /usr/bin/env python3

import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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



if __name__ == '__main__':
    rough()