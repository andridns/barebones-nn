# -*- coding: utf-8 -*-
import numpy as np

def standard_scaler(X):
    """Scale features into a unit Gaussian."""
    return (X-np.mean(X, axis=0))/np.std(X, axis=0)
    
def one_hot_encoding(v):
    """One-hot encoding for categorical features."""
    return np.eye(len(np.unique(v)))[v]

def train_test_split(X, y, test_size, random_state=42):
    """Splits X and y into training and testing sets."""
    np.random.seed(random_state) # for reproducibility
    shuffled_idx = np.random.permutation(X.shape[0]) # shuffling row indices
    X, y = X[shuffled_idx], y[shuffled_idx] # reassign X and y with shuffled indices
    n_test = int(X.shape[0] * test_size) # no of test datapoints
    X_train, X_test = X[:-n_test], X[-n_test:] # split X into X_train and X_test 
    y_train, y_test = y[:-n_test], y[-n_test:] # split y into y_train and y_test

    return X_train, X_test, y_train, y_test
