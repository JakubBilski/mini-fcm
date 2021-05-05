import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from datetime import datetime
from loadingData import univariateDatasets, loadSktime
from transformingData import derivatives, normalizing
from examiningData import displaying
import pathlib
import os

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def load_preprocessed_data(
    test_path,
    train_path,
    derivative_order):

    test_xses_series, test_ys = loadSktime.load_sktime(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = loadSktime.load_sktime(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)
    
    return train_xses_series, train_ys, test_xses_series, test_ys



if __name__ == "__main__":
    os.mkdir(plots_dir)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    plot_xs = []
    plot_ys_1 = []
    plot_ys_2 = []
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())[0:15]
    for dataset_name in datasets:
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)
        eps = 0.01
        nos_unique = []
        for xs in train_xses_series:
            unique_xs = []
            for x in xs:
                found = False
                for ux in unique_xs:
                    if sum(abs(np.asarray(x)-np.asarray(ux))) < eps:
                        found = True
                        break
                if not found:
                    unique_xs.append(x)
            nos_unique.append(len(unique_xs))
        print(f'{dataset_name}')
        print(f'min: {min(nos_unique)}')
        print(f'mean: {sum(nos_unique) / len(nos_unique)}')
        plot_xs.append(dataset_name)
        plot_ys_1.append(min(nos_unique))
        plot_ys_2.append(sum(nos_unique) / len(nos_unique))
    ax.bar(plot_xs, plot_ys_2, label='Mean number of unique points')
    ax.bar(plot_xs, plot_ys_1, label='Min number of unique points')
    ax.legend()
    ax.set_title(f"Unique points in series for epsilon = {eps}")
    ax.set_xticklabels(datasets, rotation = 45)
    plt.savefig(plots_dir / 'unique_points.png')
