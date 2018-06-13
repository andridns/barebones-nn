import numpy as np

def cat_xentropy_loss(probs, y):
    """Returns categorical cross-entropy loss."""
    return np.mean(-np.log(probs[np.arange(y.shape[0]), y]))

def accuracy(preds, y):
    """Returns accuracy score given prediction probabilities and labels"""
    return np.sum(preds==y) / float(len(y))
