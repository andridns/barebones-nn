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

def model_predict(X, y, model):
    """Obtain model prediction given X, y and model dictionary.
    
    Args:
        X: data set of examples you would like to label
        y (int): label vector of size [None,]
        model: trained model dictionary
    
    Returns:
        preds (int): prediction vector
        acc (float): accuracy score
    """
    # Forward propagation (no dropout because the model is in inference mode)
    probs, _ = model_forward(X, model['params'], model['activation'], dropout_rate=None)
    preds = np.argmax(probs, axis=1) # get predictions
    acc = accuracy(preds, y) # calculate accuracy
    print('Accuracy: {}'.format(acc))
    return preds, acc

def update_params(params, grads, learning_rate):
    """Update model parameters through gradient descent.
    
    Args:
        params: dictionary of model weights (W) and biases (b) in all layers 
        grads: dictionary of gradient parameters dA (activations), dW (weights) and db (biases)
    
    Returns:
        params: dictionary of updated model weights (W) and biases (b) in all layers 
    """
    L = len([k for k in params.keys() if 'W' in k]) # number of layers in the neural network
    for l in range(L): # parameters updates by update rule
        params["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)] # weights update
        params["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)] # biases update
        
    return params

def global_param_init(layer_dims, weight_init):
    """Initialize weight and biases parameters in the network.

    Args:
        layer_dims: list containing number of input, hidden and output neurons in sequence
        weight_init (str): weight initialization method
    
    Returns:
        params: dictionary of model weights (W) and biases (b) in all layers
    """
    np.random.seed(42)
    params = {}
    L = len(layer_dims) # no of layers
    for l in range(1, L):
        params['b' + str(l)] = np.zeros((1, layer_dims[l])) # bias initialization
        # typical weight initialization methods
        if weight_init.lower() == 'glorot_uniform':
            limit = np.sqrt(6. / (layer_dims[l-1] + layer_dims[l]))
            params['W' + str(l)] = np.random.uniform(-limit, limit, (layer_dims[l-1], layer_dims[l]))
        elif weight_init.lower() == 'glorot_normal':
            stddev = np.sqrt(2. / (layer_dims[l-1] + layer_dims[l]))
            params['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * stddev
        elif weight_init.lower() == 'he_uniform':
            limit = np.sqrt(6. / layer_dims[l-1])
            params['W' + str(l)] = np.random.uniform(-limit, limit, (layer_dims[l-1], layer_dims[l]))
        elif weight_init.lower() == 'he_normal':
            stddev = np.sqrt(2. / layer_dims[l-1])
            params['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * stddev
                                   
    return params

def log_csv(losses, train_accs, val_accs):
    """Record training data into a pandas dataframe then save it as a CSV file."""
    training_data = np.concatenate([np.array(losses).reshape(-1,1),
                                    np.array(train_accs).reshape(-1,1),
                                    np.array(val_accs).reshape(-1,1)], axis=1)
    training_df = pd.DataFrame(training_data, columns=['loss','training_accuracy','validation_accuracy'])
    training_df.index.name = 'global_step'
    training_df.to_csv('logs.csv')
    print('Training Logs are saved as a CSV file.')
    
def plot(losses, val_accs, learning_rate):
    """Plots model loss and accuracy curves and save results as png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12, 4))
    ax1.plot(np.arange(len(losses)), np.squeeze(losses))
    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve (Learning rate= {})'.format(learning_rate))
    ax2.plot(np.arange(len(val_accs)), np.squeeze(val_accs))
    ax2.set_xlabel('Global Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy (Learning rate= {})'.format(learning_rate))
    fig.savefig('results.png')
    print('Training Curves are saved as a PNG file.')
