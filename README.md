# barebones-nn

## Installation

Python version: 3+ (3.6+ is recommended)

(Optional) Setting up virtualenv is recommended:
``` bash
virtualenv -p $(which python3.6) ~/${ENV_NAME}
source ~/${ENV_NAME}/bin/activate
```

If python 3.6+ is not available:
``` bash
virtualenv -p $(which python3) ~/${ENV_NAME}
source ~/${ENV_NAME}/bin/activate
```

Where `${ENV_NAME}` is the virtual environment name.

To install dependencies:
``` bash
pip3 install -r requirements.txt
pip3 install jupyter notebook jupyterlab
```


## Sample Usage from Command-Line Interface

To build and train a feedforward network with a hidden layers containing 20 neurons:
``` bash
# Run this from root project directory
python3 train.py --ndims 4 20 3 --activation sigmoid --lrate 0.0145 --dropout 0.1
```

Similarly, for a feedforward network with 2 hidden layers, with 32 and 24 neurons respectively:
``` bash
# Run this from root project directory
python3 train.py --ndims 4 32 24 3 --activation relu --lrate 0.0145 --dropout 0.5
```

`lrate` is the model learning rate.

`dropout` is the dropout rate of the model.

## How It works

Detailed explanations of how this repository works is contained in [quickstart.ipynb](./quickstart.ipynb).

A cached HTML version of the notebook is also available in [quickstart.html](./quickstart.html).

## Folder Structure

This simple neural network library is built in a modular approach. The python modules are organized as follows:

 * [model.py](./model.py) - main model declaration
 * [layers.py](./layers.py) - layers and entire model forward and backward APIs
 * [activations.py](./activations.py) - activation gates forward and backward APIs
 * [metrics.py](./metrics.py) - loss function and evaluation metrics
 * [nn_utils.py](./nn_utils.py) - weight initializer, weight updater, prediction functions, plotting function
 * [utils.py](./utils.py) - preprocessing functions (standard scaler, train-test split)
 * [train.py](./train.py) - command-line training executable script
 * [demo.ipynb](./demo.ipynb) - main Jupyter notebook with explanations

## Further Model Customization

While the `argparser` in `train.py` only include 4 customizable parameters, it is possible to build a more detailed model configuration. Please refer to [demo.ipynb](./demo.ipynb) for a more concrete demonstration.

There are 7 customizable parameters:

 * layer_dims - layer dimensions including input and output layer
 * activation - activation function for hidden layers
 * weight_init - weight initialization method
 * dropout rate - probability of randomly setting activation to zero
 * learning_rate - scalar for gradient descent update rule
 * num_steps - number of optimization iteration (epoch)
 * early_stopping - if True, stop training iterations when maximum validation accuracy is achieved

## Output Files

Upon training completion, there are output files that might be useful for documentation purposes:
 * [logs.csv](./logs.csv) - records of training loss, training accuracy and validation accuracy
 * [results.png](./results.png) - matplotlib output for training loss and validation accuracy

## Available Activations

- [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
- [ReLU](https://arxiv.org/abs/1803.08375)
- [Softmax](https://en.wikipedia.org/wiki/Softmax_function)

## Available Weight Initializations

- [Glorot Uniform Initialization](https://keras.io/initializers/#glorot_uniform)
- [Glorot Normal Initialization](https://keras.io/initializers/#glorot_normal)
- [He Uniform Initialization](https://keras.io/initializers/#he_uniform)
- [He Normal Initialization](https://keras.io/initializers/#he_normal)

## Available Regularizations

- [Dropout](https://arxiv.org/abs/1207.0580)

## Dependencies

Dependencies used:

*   Numpy
*   Pandas
*   Matplotlib
*   tqdm
*   Scikit-learn
*   Jupyter Notebook / Jupyter Lab

Scikit-learn is only needed to download the iris dataset. Pandas and Matplotlib are only required for logging the training information into a CSV file, as well saving training curve visualization into a PNG file. Tqdm is for printing training progress bar.

All core neural network modules and data preprocessing functions are written in pure Numpy from scratch.

## Other Remarks

This library is meant for educational purposes as each module is built with minimal dependencies to external libraries except Numpy.

Please don't hesitate to contact me if some of the instructions didn't work.

Questions, comments and constructive feedbacks are thoroughly welcome :)

## TO-DOs

- Model saving (e.g. pickle)
- Convolution Operation
- Recurrent Modules
- L1L2 Regularizations
- Batch Normalization
- Etc...
