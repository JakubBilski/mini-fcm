# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from cognitiveMaps.hmm import HMM
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadSktime, univariateDatasets
from examiningData import displaying

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def create_hmms_for_classes(
    train_xses_series_by_ys,
    no_classes,
    no_states):
    hmm_by_ys = [HMM(no_states) for _ in range(no_classes)]
    for y in tqdm(range(no_classes)):
        hmm_by_ys[y].train(train_xses_series_by_ys[y], 100)
        hmm_by_ys[y].set_class(y)

    return hmm_by_ys


def load_all_data(
    data_loading_function,
    test_path,
    train_path,
    derivative_order):

    print("Loading data")
    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    print("Normalizing")
    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    return train_xses_series, train_ys, test_xses_series, test_ys


def generate_centers(
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path,
    no_centers):

    print("Learning centers with cmeans")
    centers, transformed_train_xses_series = cmeans.find_centers_and_transform(train_xses_series, no_centers)
    subtitles = [f'class {y}' for y in range(no_classes)]
    subtitles.append('cmeans')

    train_xses_series_by_ys = [[] for _ in range(no_classes)]
    for xs, y in zip(train_xses_series, train_ys):
        train_xses_series_by_ys[int(y)].append(xs)

    print("Learning hmms")
    hmms = create_hmms_for_classes(
        train_xses_series_by_ys,
        no_classes,
        no_centers
    )
    hmm_centers = [hmm.get_gauss_means() for hmm in hmms]
    hmm_covars = [hmm.get_gauss_covars() for hmm in hmms]

    train_xses_series_by_ys.append(train_xses_series)
    print("Printing results")
    displaying.display_hmm_and_cmeans_centers(
        train_xses_series_by_ys,
        plots_dir / f'{dataset_name} centers.png',
        f'{dataset_name} centers',
        subtitles,
        [*hmm_centers, centers],
        hmm_covars
    )

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file, delimiter=';')
    csv_writer.writerow(['dataset', 'no_classes', 'no_centers'])
    csv_writer.writerow([dataset_name, no_classes, no_centers])
    csv_writer.writerow(['y', 'centers'])

    for y in range(no_classes):
        to_write = [f'hmm_{y}']
        to_write.extend(str(centers) for centers in hmm_centers[y])
        csv_writer.writerow(to_write)
    csv_writer.writerow(["cmeans"] + centers)
    csv_results_file.close()

if __name__ == "__main__":

    os.mkdir(plots_dir)

    datasets = univariateDatasets.DATASETS_NAMES_WITH_NUMBER_OF_CLASSES

    no_centers = 2

    for dataset_name, no_classes in datasets:
        csv_path = f'plots\\picked\\centers_generated_with_hmm\\{dataset_name}_2_hmm_centers.csv'

        print(f"{dataset_name}")
        print(f"no_centers: {no_centers}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_all_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)
        generate_centers(
            train_xses_series=train_xses_series,
            train_ys=train_ys,
            no_classes=no_classes,
            dataset_name=dataset_name,
            csv_results_path=csv_path,
            no_centers=no_centers)

