from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt



def load_cricket(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = [[[
        float(series[variable][vector]) for variable in range(6)]
        for vector in range(len(series[0]))]
        for series in df['relationalAtt']]
    ys = [float(a) for a in df['classAttribute']]
    return xses_series, ys

def load_cricket_normalized(path):
    xses_series, ys = load_cricket(path)
    min_x = -8
    max_x = 11
    normalized = [[[(x-min_x)/(max_x-min_x) for x in vector] for vector in series] for series in xses_series]
    return normalized, ys


def load_uwave(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = [[[
        float(series[variable][vector]) for variable in range(3)]
        for vector in range(len(series[0]))]
        for series in df['relationalAtt']]
    ys = [float(a) for a in df['classAttribute']]
    return xses_series, ys

def load_uwave_normalized(path):
    xses_series, ys = load_uwave(path)
    xses_flat = [x for series in xses_series for vector in series for x in vector]
    min_x = -4
    max_x = 5
    normalized = [[[(x-min_x)/(max_x-min_x) for x in vector] for vector in series] for series in xses_series]
    return normalized, ys