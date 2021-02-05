from scipy.io import arff
import pandas as pd


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
    normalized = [[[
        (x-min_x)/(max_x-min_x)
        for x in vector]
        for vector in series]
        for series in xses_series]
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
    min_x = -4
    max_x = 5
    normalized = [[[
        (x-min_x)/(max_x-min_x) for x in vector]
        for vector in series]
        for series in xses_series]
    return normalized, ys


def load_basic_motions(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = [[[
        float(series[variable][vector]) for variable in range(6)]
        for vector in range(len(series[0]))]
        for series in df['relationalAtt']]
    classes = {b"Standing": 0, b"Running": 1, b"Walking": 2, b"Badminton": 3}
    ys = [float(classes[a]) for a in df['activity']]
    return xses_series, ys
