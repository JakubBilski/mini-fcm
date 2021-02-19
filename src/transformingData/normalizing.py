import numpy as np
import copy


def transform(xses_series, mins, maxs):
    input_size = len(xses_series[0][0])
    lens = [a - b for a, b in zip(maxs, mins)]
    normalized = [[[
        0.001+0.998*(x[variable]-mins[variable])/lens[variable]
        for variable in range(input_size)]
        for x in xs]
        for xs in xses_series]
    return normalized


def get_mins_and_maxs(xses_series):
    input_size = len(xses_series[0][0])
    extracted_xs_by_variable = [[x[variable] for xs in xses_series for x in xs]
                    for variable in range(input_size)]
    mins = [min(exs) for exs in extracted_xs_by_variable]
    maxs = [max(exs) for exs in extracted_xs_by_variable]
    return mins, maxs