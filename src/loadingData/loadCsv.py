import numpy as np
import pandas as pd



def load_csv(path):
    dataset = pd.read_csv(path)
    return dataset.head(10)