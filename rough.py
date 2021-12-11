#! /usr/bin/env python3

import sklearn
from sklearn.datasets import load_diabetes
import numpy as np

def rough():
    X, y = load_diabetes(return_X_y=True)
    print(X.shape)
    print(y.shape)
    print(X[0])
    print(y[0])

if __name__ == '__main__':
    rough()