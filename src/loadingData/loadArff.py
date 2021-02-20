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


def load_eigen_worms(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['eigenWormMultivariate_attribute'], 6)
    ys = [float(a)-1.0 for a in df['target']]
    return xses_series, ys

    
def load_epilepsy(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 3)
    classes = {b"EPILEPSY": 0, b"WALKING": 1, b"RUNNING": 2, b"SAWING": 3}
    ys = [float(classes[a]) for a in df['activity']]
    return xses_series, ys


def load_ethanol_concentration(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 3)
    classes = {b"E35": 0, b"E38": 1, b"E40": 2, b"E45": 3}
    ys = [float(classes[a]) for a in df['classValues']]
    return xses_series, ys


def load_ering(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['eRingDMultivariate_attribute'], 4)
    ys = [float(a)-1.0 for a in df['target']]
    return xses_series, ys


def load_finger_movements(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 28)
    classes = {b"left": 0, b"right": 1}
    ys = [float(classes[a]) for a in df['hand']]
    return xses_series, ys


def load_hand_movement_direction(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 10)
    classes = {b"right": 0, b"forward": 1, b"left": 2, b"backward": 3}
    ys = [float(classes[a]) for a in df['classValue']]
    return xses_series, ys


def load_handwriting(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 3)
    ys = [float(a)-1.0 for a in df['classAttribute']]
    return xses_series, ys


def load_libras(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 2)
    ys = [float(a)-1.0 for a in df['class']]
    return xses_series, ys


def load_natops(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 24)
    ys = [float(a)-1.0 for a in df['classAttribute']]
    return xses_series, ys


def load_pen_digits(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['input'], 2)
    ys = [float(a) for a in df['class']]
    return xses_series, ys


def load_phoneme(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['Phoneme'], 11)
    classes = {
        b"AA": 0,
        b"AE": 1,
        b"AH": 2,
        b"AO": 3,
        b"AW": 4,
        b"AY": 5,
        b"B": 6,
        b"CH": 7,
        b"D": 8,
        b"DH": 9,
        b"EH": 10,
        b"ER": 11,
        b"EY": 12,
        b"F": 13,
        b"G": 14,
        b"HH": 15,
        b"IH": 16,
        b"IY": 17,
        b"JH": 18,
        b"K": 19,
        b"L": 20,
        b"M": 21,
        b"N": 22,
        b"NG": 23,
        b"OW": 24,
        b"OY": 25,
        b"P": 26,
        b"R": 27,
        b"S": 28,
        b"SH": 29,
        b"T": 30,
        b"TH": 31,
        b"UH": 32,
        b"UW": 33,
        b"V": 34,
        b"W": 35,
        b"Y": 36,
        b"Z": 37,
        b"ZH": 38}
    ys = [float(classes[a]) for a in df['target']]
    return xses_series, ys


def load_racket_sports(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 6)
    classes = {b"Badminton_Smash": 0, b"Badminton_Clear": 1, b"Squash_ForehandBoast": 2, b"Squash_BackhandBoast": 3}
    ys = [float(classes[a]) for a in df['activity']]
    return xses_series, ys


def load_self_regulation_scp1(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 6)
    classes = {b"negativity": 0, b"positivity": 1}
    ys = [float(classes[a]) for a in df['cortical']]
    return xses_series, ys


def load_self_regulation_scp2(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['relationalAtt'], 7)
    classes = {b"negativity": 0, b"positivity": 1}
    ys = [float(classes[a]) for a in df['cortical']]
    return xses_series, ys


def load_spoken_arabic_digits(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['MFCCcoefficient'], 13)
    ys = [float(a)-1.0 for a in df['class']]
    return xses_series, ys


def load_stand_walk_jump(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    xses_series = _load_arff_basic_xses_series(df['ECG_Activites'], 4)
    classes = {b"standing": 0, b"walking": 1, b"jumping": 1}
    ys = [float(classes[a]) for a in df['target']]
    return xses_series, ys
