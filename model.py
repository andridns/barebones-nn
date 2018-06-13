# -*- coding: utf-8 -*-
import numpy as np
import tqdm
from layers import model_forward, model_backward
from utils import predict, global_param_init, update_params, plot, log_csv
from metrics import cat_xentropy_loss

def feedforward_neuralnet(X_train, y_train, X_test, y_test, layer_dims,
                          activation='sigmoid', weight_init='glorot_uniform',
                          dropout_rate=None, learning_rate=0.0495, num_steps=3000, 
                          early_stopping=True):
    """Build, train and evaluate feed-forward neural network model.
    
    Args:
        X_train: training input matrix of size [None, n_features]
        X_test: testing input matrix of size [None, n_features]
        y_train (int): training label vector of size [None,]
        y_test (int): testing label vector of size [None,]
        layer_dims: list containing number of input, hidden and output neurons in sequence
        activation (str): activation function for hidden layers
        weight_init (str): weight initialization method
        dropout_rate: probability of randomly setting activations to zero
        learning_rate: learning rate for gradient descent update rule
        num_steps: number of optimization iteration (epoch)
    
    Returns:
        model: dictionary containing these information as key value pairs:
            model['params'] = dictionary of model weights (W) and biases (b) in all layers
            model['losses'] = recorded loss array during training
            model['accuracy'] = recorded accuracy array during training
            model['layer_dims'] = list containing number of input, hidden and output neurons
            model['activation'] = activation function for hidden layers
            model['weight_init'] = weight initialization method
            model['dropout_rate'] = probability of randomly setting activations to zero
            model['learning_rate'] = learning rate for gradient descent update rule
            model['best_epoch'] = epoch number at highest achieved accuracy
    """
    model = {} # model will be returned as dictionary
    losses = [] # initialize loss array
    train_accs = [] # initialize training accuracy array
    val_accs = [] # initialize validation accuracy array
    params = global_param_init(layer_dims, weight_init) # initialize weights
    t = tqdm.trange(num_steps) # tqdm object
    best_epoch = num_steps # initialize best epoch value

    # Training loop
    for i in t:
        probs, caches = model_forward(X_train, params, activation, dropout_rate) # forward propagation
        loss = cat_xentropy_loss(probs, y_train) # calculate loss
        grads = model_backward(probs, y_train, caches, activation, dropout_rate) # error backpropagation
        params = update_params(params, grads, learning_rate) # weight updates
        train_acc = predict(X_train, y_train, params, activation) # training accuracy
        val_acc = predict(X_test, y_test, params, activation) # testing accuracy
        t.set_postfix(loss=float(loss), train_acc=train_acc, val_acc=val_acc) # tqdm printing
        losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # Record training logs
        if early_stopping and val_acc > 0.99:
            best_epoch = i
            print()
            print('Early Stopping at Epoch: {}'.format(i))      
            break # stop training if maximum accuracy is achieved
    print('Training Finished')
    print('Training Accuracy Score: {:.2f}%'.format(train_acc*100))
    print('Validation Accuracy Score: {:.2f}%'.format(val_acc*100))
    
    # Record training log as pandas dataframe then save as CSV
    log_csv(losses, train_accs, val_accs)

    # Plot loss and accuracy curve
    plot(losses, val_accs, learning_rate)
    
    # Update model values
    model['params'] = params
    model['losses'] = np.array(losses)
    model['training_accuracy'] = np.array(train_accs)
    model['validation_accuracy'] = np.array(val_accs)
    model['layer_dims'] = np.array(layer_dims)
    model['activation'] = activation
    model['weight_init'] = weight_init
    model['dropout_rate'] = float(dropout_rate) if dropout_rate else 0.0
    model['learning_rate'] = float(learning_rate)
    model['best_epoch'] = int(best_epoch)

    return model
