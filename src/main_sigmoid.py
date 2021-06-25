# flake8: noqa
import pathlib
import os
import csv
import time
import argparse
import itertools
from datetime import datetime

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps import baseCognitiveMap
from transformingData import cmeans
from main import load_preprocessed_data, cross_validation_folds
from modelAnalysis import accuracyComparing
from modelAnalysis import nnPredicting
from modelAnalysis import mapsExamining
from loadingData import univariateDatasets
from savingResults import csvSettings


def test_solution(
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    dataset_name,
    no_states,
    max_iter,
    fold_no,
    additional_info,
    csv_results_path,
    mutation,
    recombination,
    popsize):
    
    solution_name = 'fcm nn'
    model_class = DECognitiveMap

    csv_results_file = open(csv_results_path, 'a', newline='')
    csv_writer = csv.writer(csv_results_file)

    execution_start_timestamp = time.time()
    centers, transformed_train_xses_series = cmeans.find_centers_and_transform(
        xses_series=train_xses_series,
        c=no_states)
    transformed_test_xses_series = cmeans.transform(
        xses_series=test_xses_series,
        centers=centers)
    cmeans_execution_time = time.time() - execution_start_timestamp

    learning_input = [([xs], y) for xs, y in zip(transformed_train_xses_series, train_ys)]

    nits = []
    train_models = []
    for i in range(len(learning_input)):
        model = model_class(no_states)
        nit = model.train(learning_input[i][0], max_iter, mutation, recombination, popsize)
        nits.append(nit)
        model.set_class(learning_input[i][1])
        train_models.append(model)
    mean_nit = sum(nits) / len(nits)
    max_nit = max(nits)

    predicted = nnPredicting.predict_nn_fcm(
        train_models=train_models,
        test_xs=transformed_test_xses_series
    )

    accuracy = accuracyComparing.get_accuracy(predicted, test_ys)
    mcc = accuracyComparing.get_mcc(predicted, test_ys)

    complete_execution_time = time.time() - execution_start_timestamp
    
    degenerated_share = mapsExamining.get_share_of_degenerated_weights(train_models, threshold=0.99)

    row = [
        dataset_name,
        solution_name,
        fold_no,
        additional_info,
        no_states,
        max_iter,
        accuracy,
        mcc,
        degenerated_share,
        mean_nit,
        max_nit,
        complete_execution_time,
        cmeans_execution_time,
        '?',
        '?',
        mutation,
        recombination,
        popsize]
    csv_writer.writerow(row)
    print(row)
    csv_results_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Running ')
    parser.add_argument('--dataset_id', '-d', required=True, choices=range(0,85), type=int)
    parser.add_argument('--resultsdir', '-rd', required=False, type=str, default=f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}')
    parser.add_argument('--skipfile', '-sf', required=False, type=str)
    parser.add_argument('--fold', '-f', required=False, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.skipfile is not None:
        skip_file = pathlib.Path(args.skipfile)
        skip_file = open(skip_file, 'r', newline='')
        csv_reader = csv.reader(skip_file)
        already_tested_rows = [row for row in csv_reader]
    else:
        already_tested_rows = []

    plots_dir = pathlib.Path('plots', args.resultsdir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    csv_results_path=plots_dir / f'classification_results.csv'
    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(csvSettings.get_header())
    csv_results_file.close()

    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    datasets = [datasets[args.dataset_id]]

    tested_taus = [1, 2, 3, 5, 7]
    tested_nums_states = [3, 4, 5, 6, 7]
    tested_max_iters = [150]
    tested_mutations = [0.5]
    tested_recombinations = [0.5]
    tested_popsizes = [10]

    parameters = list(itertools.product(
        tested_taus,
        tested_nums_states,
        tested_max_iters,
        tested_mutations,
        tested_recombinations,
        tested_popsizes
    ))

    if args.fold is not None:
        if args.fold not in [0, 1, 2]:
            raise argparse.ArgumentTypeError('--fold must be from [0, 1, 2]')
        chosen_folds = [args.fold]
    else:
        chosen_folds = list(range(3))

    for dataset_name in datasets:
        print(f"Preprocessing {dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            test_path=pathlib.Path('data', 'Univariate_ts', f'{dataset_name}', f'{dataset_name}_TEST.ts'),
            train_path=pathlib.Path('data', 'Univariate_ts', f'{dataset_name}', f'{dataset_name}_TRAIN.ts'),
            derivative_order=1)

        folds = cross_validation_folds(train_xses_series, train_ys, 3)

        for tau, no_states, max_iter, mut, recomb, popsize in parameters:
            baseCognitiveMap.SIGMOID_TAU = tau
            for f in chosen_folds:
                was_already_tested = False
                for at_row in already_tested_rows:
                    if at_row[0] == str(dataset_name) and \
                        at_row[2] == str(f) and \
                        at_row[4] == str(no_states) and \
                        at_row[5] == str(max_iter) and \
                        at_row[15] == str(mut) and \
                        at_row[16] == str(recomb) and \
                        at_row[17] == str(popsize):
                        print(f"(sigmoid) Skipping {dataset_name} {no_states, max_iter, mut, recomb, popsize} {f}")
                        was_already_tested = True
                        break
                if was_already_tested:
                    continue
                fold_train_xses_series = folds[f][0]
                fold_train_ys = folds[f][1]
                fold_validation_xses_series = folds[f][2]
                fold_validation_ys = folds[f][3]
                test_solution(
                    fold_validation_xses_series,
                    fold_validation_ys,
                    fold_train_xses_series,
                    fold_train_ys,
                    dataset_name,
                    no_states,
                    max_iter,
                    f,
                    f'tau: {tau}',
                    csv_results_path,
                    mutation=mut,
                    recombination=recomb,
                    popsize=popsize)
