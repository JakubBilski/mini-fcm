# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from cognitiveMaps.deCognitiveMap import DECognitiveMap
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadSktime

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


if __name__ == "__main__":

    # os.mkdir(plots_dir)

    testing_fcm = False
    testing_hmm = True

    datasets = [
        ('ACSF1', 10),
        ('Adiac', 36),
        ('ArrowHead', 3),
    ]
    datasets = [datasets[0]]
    derivative_order = 1

    k = 3

    for dataset_name, no_classes in datasets:
        
        print(dataset_name)

        test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts')
        train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts')

        test_xses_series, test_ys = loadSktime.load_sktime(test_path)
        test_xses_series = derivatives.transform(test_xses_series, derivative_order)
        
        train_xses_series, train_ys = loadSktime.load_sktime(train_path)
        train_xses_series = derivatives.transform(train_xses_series, derivative_order)

        mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
        test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
        train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

        centers, train_transformed_xses_series = cmeans.find_centers_and_transform(train_xses_series, k)
        test_transformed_xses_series = cmeans.transform(test_xses_series, centers)

        # train_transformed_xses_series = train_xses_series

        decm = DECognitiveMap(k)
        decm.train(train_transformed_xses_series[0])
        print(decm.get_error(train_transformed_xses_series[0]))

        mppicm = MppiCognitiveMap(k)
        mppicm.train(train_transformed_xses_series[0])
        print(mppicm.get_error(train_transformed_xses_series[0]))

        plot_ys, plot_xs = [], []
        # for maxiter in range(1, 100, 1):
        #     decm = DECognitiveMap(k)
        #     decm.train(train_transformed_xses_series[0], maxiter)
        #     plot_ys.append(decm.get_error(train_transformed_xses_series[0]))
        #     print(plot_ys[-1])
        #     plot_xs.append(maxiter)
        # fig, ax = plt.subplots()
        # ax.plot(plot_xs, plot_ys)
        # ax.set(
        #     xlabel='maxiter',
        #     ylabel='err',
        #     title=f'learning with de')
        # plt.show()
        # plt.close()

        decmerrs = []
        mppicmerrs = []
        for xs in tqdm(train_transformed_xses_series):
            decm = DECognitiveMap(k)
            decm.train(xs)
            decmerrs.append(decm.get_error(xs))
            print("decm")
            print(decm.weights)

            mppicm = MppiCognitiveMap(k)
            mppicm.train(xs)
            mppicmerrs.append(mppicm.get_error(xs))
            print("mppicm")
            print(mppicm.weights)

        fig, ax = plt.subplots()
        plot_xs = range(0, len(train_transformed_xses_series))
        ax.plot(plot_xs, decmerrs, color='blue', label='decm')
        ax.plot(plot_xs, mppicmerrs, color='red', label='mppicm')
        ax.set(
            xlabel='maxiter',
            ylabel='err',
            title=f'learning with de')
        ax.legend()
        plt.show()


