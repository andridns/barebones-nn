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

Scikit-learn is only needed to download the iris dataset. Pandas and Matplotlib are not essential, but I (got carried away so I) decided to log the training information into a CSV file, as well as the training curve visualization into a PNG file. Tqdm is for printing training progress bar.

All core neural network modules and data preprocessing functions are written in pure Numpy from scratch.
