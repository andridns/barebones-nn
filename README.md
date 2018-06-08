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

## Folder Structure

I build the neural network in a modular approach. In the root project directory you'll find these python modules, each serving unique function groups:

 * [model.py](./model.py) - main model declaration
 * [layers.py](./layers.py) - layers and entire model forward and backward APIs
 * [activations.py](./activations.py) - activation gates forward and backward APIs
 * [metrics.py](./metrics.py) - loss function and evaluation metrics
 * [nn_utils.py](./nn_utils.py) - weight initializer, weight updater, prediction functions, plotting function
 * [utils.py](./utils.py) - preprocessing functions (standard scaler, train-test split)
 * [train.py](./train.py) - command-line training executable script
 * [demo.ipynb](./demo.ipynb) - main Jupyter notebook with explanations

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

