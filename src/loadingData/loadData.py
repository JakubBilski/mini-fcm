import numpy as np
import pandas as pd



def load_acute(data_path):
    return pd.read_csv(data_path, header=None, delim_whitespace=True, encoding='UTF-16')

    
def load_acute_normalized(data_path):
    df = load_acute(data_path)
    df[0] = (df[0].str.replace(',', '.').astype(float) - 35.0) / (42.0 - 35.0)
    for i in range(1,8):
        df[i] = df[i].replace({'yes': 1, 'no': 0})
    return df
