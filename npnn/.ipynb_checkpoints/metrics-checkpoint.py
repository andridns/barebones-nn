import numpy as np

def cat_xentropy_loss(probs, y):
    """Returns categorical cross-entropy loss."""
    return np.mean(-np.log(probs[np.arange(y.shape[0]), y]))

def accuracy(preds, y):
    """Returns accuracy score given prediction probabilities and labels."""
    return np.sum(preds==y) / float(len(y))

def mean_squared_error(preds, y):
    """Returns mean squared error (MSE) between predictions and labels."""
    return np.mean((y-preds)**2)

def mean_absolute_error(preds, y):
    """Returns mean absolute error (MAE) between predictions and labels."""
    return np.mean(np.abs(y-preds))

def mean_absolute_percentage_error(preds, y):
    """Returns mean absolute percentage error (MAPE) between predictions and labels."""
    diff = np.abs((y-preds) / (y+1e-8))
    return 100. * np.mean(diff)
