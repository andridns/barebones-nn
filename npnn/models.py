import sys
sys.path.append('../')
from npnn.utils import global_param_init

def feedforward(layer_dims,
                activation='relu', weight_init='glorot_uniform',
                learning_rate = 0.001, dropout_rate=0.0):
    model = {}
    model['layer_dims'] = layer_dims
    model['activation'] = activation.lower()
    model['learning_rate'] = learning_rate
    model['weight_init'] = weight_init.lower()
    model['dropout_rate'] = dropout_rate
    model = global_param_init(model)
    
    return model