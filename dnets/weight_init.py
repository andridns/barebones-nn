import numpy as np

def glorot_uniform(n_in, n_out):
    limit = np.sqrt(6. / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def glorot_normal(n_in, n_out):
    stddev = np.sqrt(2. / (n_in + n_out))
    return np.random.randn(n_in, n_out) * stddev

def he_uniform(n_in, n_out):
    limit = np.sqrt(6. / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))

def he_normal(n_in, n_out):
    stddev = np.sqrt(2. / n_in)
    return np.random.randn(n_in, n_out) * stddev