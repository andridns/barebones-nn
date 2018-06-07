# -*- coding: utf-8 -*-
import numpy as np

def standard_scaler(X):
    """Scale features into a unit Gaussian."""
    return (X-np.mean(X, axis=0))/np.std(X, axis=0)
    
def one_hot_encoding(v):
    """One-hot encoding for categorical features."""
    return np.eye(len(np.unique(v)))[v]
