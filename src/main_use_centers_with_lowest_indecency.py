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

def generate_random_centers(no_samples, no_centers):
    np.random.seed(1)
    centerss = [np.asarray([np.random.rand(2) for _1 in range(no_centers)]) for _ in range(no_samples)]
    return centerss


def use_centers_to_train_models(train_xses_series, train_ys, centers):
    print(f'\nExecuting for centers:')
    print(centers)
    print(f'Transforming to the centers\' space')

    train_xses_series_transformed = cmeans.transform(
        xses_series=train_xses_series,
        centers=centers)

    train_models = []

    print(f'Learning train')
    for xs, y in tqdm(zip(train_xses_series_transformed, train_ys)):
        model = DECognitiveMap(no_centers)
        model.train(xs)
        model.set_class(y)
        train_models.append(model)

    print("Example model:")
    print(train_models[0].weights)

    print("Share of degenerated weights:")
    indecency = get_models_indecency(train_models)
    print(indecency)

    return train_models, indecency

def get_classification_score(
    train_models,
    test_xses_series,
    test_ys,
    centers):

    print(f'Transforming test data to the centers\' space')

    test_xses_series_transformed = cmeans.transform(test_xses_series, centers)
    test_models = []

    for xs, y in zip(test_xses_series_transformed, test_ys):
        model = DECognitiveMap(no_centers)
        model.set_class(y)
        test_models.append(model)

    print(f'classifying with best_prediction')
    accuracy = accuracyComparing.get_accuracy(
        train_models=train_models,
        test_models=test_models,
        test_xs=test_xses_series_transformed,
        input_size=no_centers,
        no_classes=no_classes,
        classification_method="best_prediction")
    print(f'accuracy: {accuracy}')
    return accuracy


def get_models_indecency(models):
    threshold = 0.99
    no_degenerated_weights = 0
    for model in models:
        no_degenerated_weights += np.sum(model.weights >= threshold)
        no_degenerated_weights += np.sum(model.weights <= -threshold)
    return no_degenerated_weights/(models[0].n*models[0].n*len(models))


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


if __name__ == "__main__":

    os.mkdir(plots_dir)

    datasets = univariateDatasets.DATASETS_NAMES_WITH_NUMBER_OF_CLASSES
    datasets = [datasets[0]]

    range_centers = [2,3]
    no_random_samples = 1

    for dataset_name, no_classes in datasets:
        csv_path = plots_dir / f'{dataset_name}_results.csv'
        csv_results_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_results_file)
        csv_writer.writerow(['dataset', 'no_classes', 'method', 'no_random_samples'])
        csv_writer.writerow([dataset_name, no_classes, 'random center with lowest indecency', no_random_samples])
        csv_writer.writerow(['no_centers', 'accuracy'])
        print(f"{dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_all_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)

        for no_centers in range_centers:
            print(f"\nno_centers: {no_centers}")
            random_centerss = generate_random_centers(no_random_samples, no_centers)
            chosen_models = None
            chosen_centers = None
            chosen_models_indecency = 1.01

            for random_centers in random_centerss:
                train_models, ind = use_centers_to_train_models(
                    train_xses_series,
                    train_ys,
                    random_centers
                )
                if ind < chosen_models_indecency:
                    chosen_models = train_models
                    chosen_centers = random_centers
                    chosen_models_indecency = ind
                    if chosen_models_indecency == 0:
                        print("Found solution with indecency 0.0")
                        break
            
            print("\nGenerating cmeans centers")

            accuracy = get_classification_score(
                    chosen_models,
                    test_xses_series,
                    test_ys,
                    chosen_centers
                )
            
            csv_writer.writerow([no_centers, accuracy])
        csv_results_file.close()
