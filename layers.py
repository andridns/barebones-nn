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

def layer_forward(A_prev, W, b, activation, dropout_rate=None):
    """Forward propagation of a dense layer with an activation function of choice.

    Args:
        A_prev: activation matrix from previous layer
        W: weight matrix of current layer 
        b: bias vector of current layer
        activation (str): activation function, e.g. "sigmoid", "relu"
        dropout_rate (float): probability of randomly setting activations to zero

    Returns:
        A: output of the activation function
        cache: tuple of values needed for backpropagation
    """
    if activation.lower() == "sigmoid":
        Z, linear_cache = dense_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        if dropout_rate: # apply dropout if not None
            A, dropout_mask = dropout(A, dropout_rate)
            cache = (linear_cache, activation_cache, dropout_mask)
    elif activation.lower() == "relu":
        Z, linear_cache = dense_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        if dropout_rate: # apply dropout if not None
            A, dropout_mask = dropout(A, dropout_rate)
            cache = (linear_cache, activation_cache, dropout_mask)

    return A, cache

def layer_backward(dA, cache, activation, dropout_rate=None):
    """Backpropagation of a dense layer with an activation function of choice.

    Args:
        dA: gradient matrix w.r.t activations in current layer
        cache: tuple of values recorded during forward propagation
        activation (str): activation function, e.g. "sigmoid", "relu"
        dropout_rate (float): probability of randomly setting activations to zero

    Returns:
        dA_prev: gradient matrix w.r.t activations in previous layer
        dW: gradient matrix w.r.t weights in current layer
        db: gradient matrix w.r.t biases in current layer
    """
    if dropout_rate:
        linear_cache, activation_cache, dropout_mask = cache
    else:
        linear_cache, activation_cache = cache
    
    if activation.lower() == "relu":
        if dropout_rate: # scale up activations if dropout is used
            dA = dropout_backward(dA, dropout_mask, dropout_rate)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = dense_backward(dZ, linear_cache)
    elif activation.lower() == "sigmoid":
        if dropout_rate: # scale up activations if dropout is used
            dA = dropout_backward(dA, dropout_mask, dropout_rate)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = dense_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def model_forward(X, params, activation, dropout_rate=None):
    """Model forward propagation from inputs to prediction probabilities.
    
    Args:
        X: input matrix of size [None, n_features]
        params: dictionary of model weights (W) and biases (b) in all layers 
        activation (str): activation function, e.g. "sigmoid", "relu"
        dropout_rate (float): probability of randomly setting activations to zero    

    Returns:
        probs (float): prediction probability vector from model forward propagation
        caches: list of tuple of values recorded during forward propagation
    """
    caches = []
    A = X
    L = len([k for k in params.keys() if 'W' in k]) # number of layers in the neural network
    # Forward propagation from first layer to the layer before softmax
    for l in range(1, L):
        A_prev = A 
        A, cache = layer_forward(A_prev, params['W' + str(l)], params['b' + str(l)], activation, dropout_rate)
        caches.append(cache)
    # Forward propagation in softmax layer
    A_prev = A
    W = params['W' + str(L)]
    b = params['b' + str(L)]
    Z, linear_cache = dense_forward(A_prev, W, b)
    probs, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    caches.append(cache) 

    return probs, caches

def model_backward(probs, y, caches, activation, dropout_rate=None):
    """Error backpropagation to all model parameters from last layer to first layer.
    
    Args:
        probs (float): prediction probability vector from model forward propagation
        y (int): label vector of size [None,]
        caches: list of tuple of values recorded during forward propagation
        activation (str): activation function, e.g. "sigmoid", "relu"
        dropout_rate (float): probability of randomly setting activations to zero    

    Returns:
        grads: dictionary of gradient parameters dA (activations), dW (weights) and db (biases)
    """
    grads = {}
    L = len(caches) # the number of layers
    m = probs.shape[0]
    # Backpropagation to softmax layer
    dZ = softmax_backward(probs, y)
    current_cache = caches[L-1]
    linear_cache, _ = current_cache
    dA_prev_temp, dW_temp, db_temp = dense_backward(dZ, linear_cache)
    grads["dA" + str(L)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    # Backpropagation from layer before softmax to first layer
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = layer_backward(grads["dA" + str(l + 2)], current_cache, activation, dropout_rate)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


