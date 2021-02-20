import numpy as np


def get_volatility_taxicab(xs):
    return np.mean(np.abs(np.asarray(xs[:-1])-np.asarray(xs[1:])))