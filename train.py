import tqdm
import numpy as np
from sklearn import datasets
from npnn.utils import train, plot, log_csv
from npnn.models import feedforward
from npnn.preprocessing import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n','--ndims', nargs='+', required=True, type=int, help='layer dimensions including input and output')
parser.add_argument('-a','--activation', type=str, default='relu', help='hidden layer activation')
parser.add_argument('-l','--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-d','--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--init', type=str, default='he_normal', help='weight initializer')
parser.add_argument('-e','--num_epoch', type=int, default=10000, help='number of training epoch')
args = parser.parse_args()

# Store model configs in a dict
model_configs = {'layer_dims':args.ndims, # layer dimensions
                 'activation':args.activation, # activation function in hidden layer
                 'weight_init':args.init, # weight initialization method
                 'dropout_rate':args.dropout, # (1-keep_prob)
                 'learning_rate':args.lr}
num_steps = args.num_epoch

# Load Dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Setup model and train
model = feedforward(**model_configs)
model = train(model, X_train, X_test, y_train, y_test, num_steps=num_steps)