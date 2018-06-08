#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from model import feedforward_neuralnet
from utils import standard_scaler, train_test_split
from sklearn import datasets

if __name__ == "__main__":
    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--ndims", nargs='+', required=True, type=int,
        help="layer dimensions including input and output")
    ap.add_argument("-a", "--activation", required=True, type=str,
        help="activation function in hidden layers")
    ap.add_argument("-l", "--lrate", type=float, default=0.0495,
        help="model learning rate")
    ap.add_argument("-d", "--dropout", type=float, default=0.5,
        help="dropout rate")
    args = ap.parse_args() # parse argument parser
    iris = datasets.load_iris() # download Iris dataset
    X, y = iris.data, iris.target # assign inputs and labels
    X = standard_scaler(X) # scale features in X to unit gaussian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10) # train-test split
    # Set up model configuration
    model_a_configs = {'X_train':X_train,
                       'y_train':y_train,
                       'X_test':X_test,
                       'y_test':y_test, 
                       'layer_dims':args.ndims, # layer dimensions
                       'activation':args.activation, # activation function in hidden layer
                       'weight_init':'glorot_uniform', # weight initialization method
                       'dropout_rate':args.dropout, # probability of randomly setting activations to 0
                       'learning_rate':args.lrate, # learning rate
                       'num_steps':10000, # num of epochs
                       'early_stopping':True} # stop training if maximum accuracy achieved
    model_a = feedforward_neuralnet(**model_a_configs) # build, train, and evaluate model

