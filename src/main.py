# flake8: noqa
import pathlib
import os
import csv
import time
from datetime import datetime

from cognitiveMaps.psoCognitiveMap import PSOCognitiveMap
from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.deShrinkedCognitiveMap import DEShrinkedCognitiveMap
from cognitiveMaps.deVeryShrinkedCognitiveMap import DEVeryShrinkedCognitiveMap
from cognitiveMaps.hmm import HMM
from loadingData import univariateDatasets
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from testingResults import mapsExamining
from loadingData import loadSktime

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def test_solution(
    solution_name,
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    no_states,
    max_iter,
    fold_no,
    additional_info,
    csv_results_path):
    
    if solution_name in ['hmm nn', 'hmm 1c']:
        model_class = HMM
    elif solution_name in ['fcm nn', 'fcm 1c']:
        model_class = DECognitiveMap
    elif solution_name == 'sfcm nn':
        model_class = DEShrinkedCognitiveMap
    elif solution_name == 'vsfcm nn':
        model_class = DEVeryShrinkedCognitiveMap
    elif solution_name == 'pso nn':
        model_class = PSOCognitiveMap
    else:
        raise Exception(f"Solution name {solution_name} not recognized")

    csv_results_file = open(csv_results_path, 'a', newline='')
    csv_writer = csv.writer(csv_results_file)

    execution_start_timestamp = time.time()
    if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn', 'pso nn']:
        centers, transformed_train_xses_series = cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_states)
        transformed_test_xses_series = cmeans.transform(
            xses_series=test_xses_series,
            centers=centers)
        cmeans_execution_time = time.time() - execution_start_timestamp
    else:
        transformed_train_xses_series = train_xses_series
        transformed_test_xses_series = test_xses_series
        cmeans_execution_time = "?"

    if solution_name in ['hmm 1c', 'fcm 1c']:
        learning_input = [([], i) for i in range(no_classes)]
        for xs, y in zip(transformed_train_xses_series, train_ys):
            learning_input[y][0].append(xs)
    else:
        learning_input = [([xs], y) for xs, y in zip(transformed_train_xses_series, train_ys)]

    error_occured = False
    nits = []
    train_models = []
    for i in range(len(learning_input)):
        model = model_class(no_states)
        try:
            nit = model.train(learning_input[i][0], max_iter)
            nits.append(nit)
        except:
            error_occured = True
            break
        model.set_class(learning_input[i][1])
        train_models.append(model)
    mean_nit = sum(nits) / len(nits)
    max_nit = max(nits)

    if not error_occured:
        try:
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

    complete_execution_time = time.time() - execution_start_timestamp
    
    if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn', 'pso nn']:
        degenerated_share = mapsExamining.get_share_of_degenerated_weights(train_models, threshold=0.99)
    else:
        degenerated_share = "?"

    if error_occured:
        print(f"error occured for no_states {no_states}. Continuing")
    else:
        row = [
            dataset_name,
            solution_name,
            fold_no,
            additional_info,
            no_states,
            max_iter,
            accuracy,
            degenerated_share,
            mean_nit,
            max_nit,
            complete_execution_time,
            cmeans_execution_time]
        csv_writer.writerow(row)
        print(row)

    csv_results_file.close()


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


def cross_validation_folds(xses_series, ys, k):
    '''performs cross-validation split while 
    ensuring the ratio of instances for each class
    remains the same'''
    indexes = [[i for i in range(len(ys)) if ys[i]==y] for y in range(max(ys)+1)]    
    f_indexes = [[i for ind in indexes for i in ind[m*len(ind)//k : (m+1)*len(ind)//k]] for m in range(k)]

    result = []
    for m in range(k):
        complement = list(set(range(len(ys)))-set(f_indexes[m]))
        train_xses_series = [xses_series[i] for i in complement]
        test_xses_series = [xses_series[i] for i in f_indexes[m]]
        train_ys = [ys[i] for i in complement]
        test_ys = [ys[i] for i in f_indexes[m]]
        result.append((train_xses_series, train_ys, test_xses_series, test_ys))
    
    return result


if __name__ == "__main__":

    tested_datasets = univariateDatasets.DATASETS_NAMES_WITH_NUMBER_OF_CLASSES

    # tested_solutions = ['sfcm nn', 'hmm nn', 'fcm 1c', 'hmm 1c', 'fcm nn', 'vsfcm nn', 'pso nn']
    tested_solutions = ['vsfcm nn', 'fcm nn', 'pso nn']

    tested_nos_states = [3, 5, 8]

    os.mkdir(plots_dir)

    csv_results_path=plots_dir / f'classification_results.csv'
    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow([
        'dataset',
        'method',
        'fold_no',
        'additional_info',
        'no_states',
        'maxiter',
        'accuracy',
        'degenerated_share',
        'mean_no_iterations',
        'max_no_iterations',
        'complete_execution_time',
        'cmeans_time'])
    csv_results_file.close()

    for dataset_name, no_classes in tested_datasets:
        print(f"Preprocessing {dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)

        folds = cross_validation_folds(train_xses_series, train_ys, 3)

        for solution_name in tested_solutions:
            max_iter = 200
            if solution_name == "pso nn":
                max_iter = 1000
            for no_states in tested_nos_states:
                for f in range(len(folds)):
                    fold_train_xses_series = folds[f][0]
                    fold_train_ys = folds[f][1]
                    fold_validation_xses_series = folds[f][2]
                    fold_validation_ys = folds[f][3]
                    test_solution(
                        solution_name,
                        fold_validation_xses_series,
                        fold_validation_ys,
                        fold_train_xses_series,
                        fold_train_ys,
                        no_classes,
                        dataset_name,
                        no_states,
                        max_iter,
                        f,
                        "",
                        csv_results_path)
