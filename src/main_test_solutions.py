# flake8: noqa
from tqdm import tqdm
import pathlib
import os
import csv
from datetime import datetime
import argparse

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.deShrinkedCognitiveMap import DEShrinkedCognitiveMap
from cognitiveMaps.deVeryShrinkedCognitiveMap import DEVeryShrinkedCognitiveMap
from cognitiveMaps.hmm import HMM
from loadingData import univariateDatasets
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadSktime
from examiningData import displaying
from examiningData import basicExamining

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def test_solution(
    solution_name,
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path):
    
    if solution_name in ['hmm nn', 'hmm 1c']:
        model_class = HMM
    elif solution_name in ['fcm nn', 'fcm 1c']:
        model_class = DECognitiveMap
    elif solution_name == 'sfcm nn':
        model_class = DEShrinkedCognitiveMap
    elif solution_name == 'vsfcm nn':
        model_class = DEVeryShrinkedCognitiveMap
    else:
        raise Exception(f"Solution name {solution_name} not recognized")

    print(f"Starting: {dataset_name} with {solution_name}")

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(['dataset', 'no_classes', 'method'])
    csv_writer.writerow([dataset_name, no_classes, solution_name + " 1it"])
    csv_writer.writerow(['no_states', 'accuracy', 'degenerated_share'])

    for no_states in range(2, 11):
        print(f"no states {no_states}")

        if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
            print(f'transforming with cmeans')
            centers, transformed_train_xses_series = cmeans.find_centers_and_transform(
                xses_series=train_xses_series,
                c=no_states)
            transformed_test_xses_series = cmeans.transform(
                xses_series=test_xses_series,
                centers=centers)
        else:
            transformed_train_xses_series = train_xses_series
            transformed_test_xses_series = test_xses_series

        if solution_name in ['hmm 1c', 'fcm 1c']:
            learning_input = [([], i) for i in range(no_classes)]
            for xs, y in zip(transformed_train_xses_series, train_ys):
                learning_input[y][0].append(xs)
        else:
            learning_input = [([xs], y) for xs, y in zip(transformed_train_xses_series, train_ys)]

        print(f'learning train models')
        error_occured = False
        train_models = []
        for i in tqdm(range(len(learning_input))):
            model = model_class(no_states)
            try:
                if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
                    model.train(learning_input[i][0], 1)
                else:
                    model.train(learning_input[i][0])
            except:
                error_occured = True
                break
            model.set_class(learning_input[i][1])
            train_models.append(model)
        
        if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
            degenerated_share = basicExamining.get_share_of_degenerated_weights(train_models, 0.99)
        else:
            degenerated_share = -1

        if not error_occured:
            try:
                print(f'classifying with best_prediction')
                if solution_name in ['hmm nn', 'hmm 1c']:
                    accuracy = accuracyComparing.get_accuracy_hmm_best_prediction(
                        train_models=train_models,
                        test_xs=transformed_test_xses_series,
                        test_classes=test_ys
                    )
                else:
                    accuracy = accuracyComparing.get_accuracy_fcm_best_prediction(
                        train_models=train_models,
                        test_xs=transformed_test_xses_series,
                        test_classes=test_ys
                    )
            except:
                error_occured = True

        if error_occured:
            print(f"error occured for no_states {no_states}")
            break
        else:
            csv_writer.writerow([no_states, accuracy, degenerated_share])
            print(f'accuracy: {accuracy}')
            print(f'share of degenerated weights: {degenerated_share}')
    
    csv_results_file.close()


def load_preprocessed_data(
    data_loading_function,
    test_path,
    train_path,
    derivative_order,
    dataset_name):

    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    displaying.display_series(
        train_xses_series,
        plots_dir / f'{dataset_name} visualization.png',
        f'{dataset_name} visualization')
    
    return train_xses_series, train_ys, test_xses_series, test_ys



def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification accuracy on TimeSeriesClassification datasets')
    parser.add_argument('--solution',
                        '-s',
                        choices=['sfcm nn', 'hmm nn', 'fcm 1c', 'hmm 1c', 'fcm nn', 'vsfcm nn'],
                        default='sfcm nn',
                        help='How models used during classification will be trained',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    solution_name = args.solution

    os.mkdir(plots_dir)

    datasets = univariateDatasets.DATASETS_NAMES_WITH_NUMBER_OF_CLASSES

    for dataset_name, no_classes in datasets:
        print(f"Preprocessing {dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1,
            dataset_name=dataset_name)

        print('_________')
        test_solution(
            solution_name=solution_name,
            test_xses_series=test_xses_series,
            test_ys=test_ys,
            train_xses_series=train_xses_series,
            train_ys=train_ys,
            no_classes=no_classes,
            dataset_name=dataset_name,
            csv_results_path=plots_dir / f'{dataset_name}_{solution_name}_classification_results.csv')
