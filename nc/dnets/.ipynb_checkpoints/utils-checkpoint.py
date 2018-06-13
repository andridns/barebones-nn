from __future__ import absolute_import
import numpy as np
from dnets import weight_init
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dnets.layers import model_forward
from dnets.metrics import accuracy

def predict(X, y, model):
    """Calculate accuracy given X, y, model parameters and activation."""
    # Forward propagation (no dropout because the model is in inference mode)
    params = model.get('var')
    act = model.get('activation')
    probs, _ = model_forward(X, model)
    preds = np.argmax(probs, axis=1) # get predictions
    acc = accuracy(preds, y) # calculate accuracy
    return acc

def model_predict(X, y, model):
    """Obtain model prediction given X, y and model dictionary."""
    # Forward propagation (no dropout because the model is in inference mode)
    probs, _ = model_forward(X, model)
    preds = np.argmax(probs, axis=1) # get predictions
    acc = accuracy(preds, y) # calculate accuracy
    print('Accuracy: {}'.format(acc))
    return preds, acc

def update_params(model, grads):
    """Update model parameters through gradient descent."""
    learning_rate = model.get('learning_rate')
    L = len([k for k in model['var'].keys() if 'W' in k]) # number of layers in the neural network
    for l in range(L): # parameters updates by update rule
        model['var']["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)] # weights update
        model['var']["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)] # biases update
        
    return model

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
    
def init_model(**kwargs):
    model = {}
    for k,v in kwargs.items():
        if type(v) == str:
            model[k] = v.lower()
        else:
            model[k] = v
    return model

def global_param_init(model, random_seed=42):
    """Initialize weight and biases parameters in the network."""
    np.random.seed(random_seed)
    initializer = model['weight_init']
    ndims = model['layer_dims']
    model['var'] = {}
    L = len(ndims) # no of layers
    for l in range(1, L):
        model['var']['b' + str(l)] = np.zeros((1, model['layer_dims'][l])) # bias initialization
        model['var']['W' + str(l)] = getattr(weight_init, f'{initializer}')(ndims[l-1], ndims[l])
        
    return model