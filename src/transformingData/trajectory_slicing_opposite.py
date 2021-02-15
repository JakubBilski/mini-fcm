import numpy as np
import copy
from numpy.core.fromnumeric import shape

from numpy.core.records import array


def transform(xses_series):
    input_size = len(xses_series[0][0])
    transformed_xses_series = [[np.ones(input_size) - x for x in xs] for xs in xses_series]

    return transformed_xses_series