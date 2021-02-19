from scipy.io import arff
import pandas as pd
import numpy as np


def _load_arff_basic_xses_series(relational_att, no_variables):
    xses_series = [[[
        float(series[variable][vector]) for variable in range(no_variables)]
        for vector in range(len(series[0]))]
        for series in relational_att]
    return xses_series


def load_cricket(path):
    data, _ = arff.loadarff(path)   
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 6)
    ys = [float(a)-1.0 for a in df['classAttribute']]
    return xses_series, ys


def load_uwave(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 3)
    ys = [float(a) for a in df['classAttribute']]
    return xses_series, ys


def load_basic_motions(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 6)
    classes = {b"Standing": 0, b"Running": 1, b"Walking": 2, b"Badminton": 3}
    ys = [float(classes[a]) for a in df['activity']]
    return xses_series, ys


def load_atrial_fibrilation(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['ECG_Atrial_Fibrilation'], 2)
    classes = {b"n": 0, b"s": 1, b"t": 2}
    ys = [float(classes[a]) for a in df['target']]
    return xses_series, ys


def load_articulary_word_recognition(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 9)
    ys = [float(a)-1.0 for a in df['classAttribute']]
    return xses_series, ys
