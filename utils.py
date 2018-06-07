# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers import model_forward
from metrics import accuracy

def predict(X, y, params, activation):
    """Calculate accuracy given X, y, model parameters and activation.
    
    Args:
        X: data set of examples you would like to label
        y (int): label vector of size [None,]
        params: dictionary of model weights (W) and biases (b) in all layers
        activation (str): activation function for hidden layers
    
    Returns:
        acc (float): accuracy score
    """
    # Forward propagation (no dropout because the model is in inference mode)
    probs, _ = model_forward(X, params, activation, dropout_rate=None)
    preds = np.argmax(probs, axis=1) # get predictions
    acc = accuracy(preds, y) # calculate accuracy
    return acc
