# -*- coding: utf-8 -*-
import numpy as np
from activations import sigmoid, sigmoid_backward
from activations import relu, relu_backward
from activations import softmax, softmax_backward

def dropout(A, dropout_rate):
    """Dropout forward propagation with dropout mask returned."""
    dropout_mask = np.random.binomial([np.ones_like(A)], 1.0 - dropout_rate)[0]
    A *= dropout_mask / (1.0 - dropout_rate)
    return A, dropout_mask

def dropout_backward(dA, dropout_mask, dropout_rate):
    """Dropout backpropagation which scales up un-dropped activation signals"""
    return dA * dropout_mask / (1.0 - dropout_rate)

def dense_forward(A, W, b):
    """Dense (fully-connected) layer forward propagation."""
    cache = (A, W, b)
    Z = A.dot(W) + b
    return Z, cache

def dense_backward(dZ, cache):
    """Dense (fully-connected) layer backpropagation."""
    A_prev, W, b = cache
    m = A_prev.shape[1] # no of datapoints
    dW = 1./m * A_prev.T.dot(dZ) # gradients w.r.t weights
    db = 1./m * np.sum(dZ, axis = 0, keepdims = True) # gradients w.r.t biases
    dA_prev = dZ.dot(W.T) # gradients w.r.t previous layer activations
    
    return dA_prev, dW, db
