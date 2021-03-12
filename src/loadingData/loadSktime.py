from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np

def load_sktime(path):
    xs, ys = load_from_tsfile_to_dataframe(path, replace_missing_vals_with='NaN')
    xses_series = xs.to_numpy().tolist()
    dims = len(xses_series[0])
    for i in range(len(xses_series)):
        xses_series[i] = [[xses_series[i][k][j] for k in range(dims)] for j in range(len(xses_series[i][0]))]

    return xses_series, ys
