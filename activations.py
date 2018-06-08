import numpy as np

def sigmoid(Z):
    """Sigmoid forward propagation layer with cache returned."""  
    cache = Z.copy()
    A = 1/(1+np.exp(-Z))
    return A, cache

def sigmoid_backward(dA, cache):
    """Sigmoid backward propagation to return gradient w.r.t pre-activation matrix."""  
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu(Z):
    """ReLU forward propagation layer with cache returned."""
    cache = Z.copy()
    A = np.maximum(0,Z)
    return A, cache

def relu_backward(dA, cache):
    """ReLU backward propagation to return gradient w.r.t pre-activation matrix.""" 
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    """Softmax forward propagation with cache returned."""
    cache = Z.copy()
    Z -= np.max(Z, axis=1, keepdims=True) # normalization trick for numerical stability
    probs = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
    return probs, cache

def softmax_backward(probs, y):
    """Softmax backward propagation to return gradient w.r.t pre-activation matrix."""
    probs[np.arange(y.shape[0]), y] -= 1
    return probs
