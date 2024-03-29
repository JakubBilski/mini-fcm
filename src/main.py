# flake8: noqa
import pathlib
import os
import csv
import time
import argparse
import itertools
from datetime import datetime

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.deShrinkedCognitiveMap import DEShrinkedCognitiveMap
from cognitiveMaps.deVeryShrinkedCognitiveMap import DEVeryShrinkedCognitiveMap
from cognitiveMaps.hmm import HMM
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from modelAnalysis import accuracyComparing
from modelAnalysis import nnPredicting
from modelAnalysis import mapsExamining
from loadingData import loadSktime
from loadingData import univariateDatasets
from savingResults import csvSettings


def test_solution(
    solution_name,
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
    no_random_initializations,
    covariance_type,
    mutation,
    recombination,
    popsize):
    
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

    csv_results_file = open(csv_results_path, 'a', newline='')
    csv_writer = csv.writer(csv_results_file)

    execution_start_timestamp = time.time()
    if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
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
        no_classes = max(train_ys) + 1
        learning_input = [([], i) for i in range(no_classes)]
        for xs, y in zip(transformed_train_xses_series, train_ys):
            learning_input[y][0].append(xs)
    else:
        learning_input = [([xs], y) for xs, y in zip(transformed_train_xses_series, train_ys)]

    nits = []
    train_models = []
    for i in range(len(learning_input)):
        model = model_class(no_states)
        if solution_name in ['hmm nn', 'hmm 1c']:
            nit = model.train(learning_input[i][0], max_iter, no_random_initializations, covariance_type)
            if nit is None:
                row = [
                    dataset_name,
                    solution_name,
                    fold_no,
                    additional_info + "failed to learn one of the models",
                    no_states,
                    max_iter,
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                    '?',
                    no_random_initializations,
                    covariance_type,
                    mutation,
                    recombination,
                    popsize]
                csv_writer.writerow(row)
                print(row)
                csv_results_file.close()
                return
        else:
            nit = model.train(learning_input[i][0], max_iter, mutation, recombination, popsize)
        nits.append(nit)
        model.set_class(learning_input[i][1])
        train_models.append(model)
    mean_nit = sum(nits) / len(nits)
    max_nit = max(nits)

    if solution_name in ['hmm nn', 'hmm 1c']:
        predicted = nnPredicting.predict_nn_hmm(
            train_models=train_models,
            test_xs=transformed_test_xses_series
        )
    else:
        predicted = nnPredicting.predict_nn_fcm(
            train_models=train_models,
            test_xs=transformed_test_xses_series
        )

    accuracy = accuracyComparing.get_accuracy(predicted, test_ys)
    mcc = accuracyComparing.get_mcc(predicted, test_ys)

    complete_execution_time = time.time() - execution_start_timestamp
    
    if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
        degenerated_share = mapsExamining.get_share_of_degenerated_weights(train_models, threshold=0.99)
    else:
        degenerated_share = "?"

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
        no_random_initializations,
        covariance_type,
        mutation,
        recombination,
        popsize]
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


def parse_args():
    parser = argparse.ArgumentParser(description='Running ')
    parser.add_argument('--dataset_id', '-d', required=True, choices=range(0,85), type=int)
    parser.add_argument(
        '--method',
        '-m',
        nargs=2,
        required=True,
        type=str)
    parser.add_argument('--resultsdir', '-rd', required=False, type=str, default=f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}')
    parser.add_argument('--skipfile', '-sf', required=False, type=str)
    parser.add_argument('--bigstates', '-b', required=False, action='store_true')
    parser.add_argument('--fold', '-f', required=False, type=int)
    args = parser.parse_args()
    args.method = f"{args.method[0]} {args.method[1]}"
    if args.bigstates and args.method != 'hmm nn' and args.method != 'fcm nn':
        raise argparse.ArgumentTypeError('--bigstates is only valid with fcm nn or hmm nn method')
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
    if args.dataset_id is not None:
        datasets = [datasets[args.dataset_id]]

    tested_methods = [args.method]

    if args.bigstates:
        tested_nums_states = [8, 9, 10, 12, 16]
    else:
        tested_nums_states = [3, 4, 5, 6, 7]

    if args.method == 'fcm nn':
        tested_nos_random_initializations = ['?']
        tested_covariance_types = ['?']
        if args.bigstates:
            tested_max_iters = [150]
            tested_mutations = [0.5]
            tested_recombinations = [0.5]
            tested_popsizes = [10]
        else:
            tested_max_iters = [150, 200, 250]
            tested_mutations = [0.5, 0.8]
            tested_recombinations = [0.5, 0.9]
            tested_popsizes = [10, 15]
    elif args.method == 'hmm nn':
        tested_nos_random_initializations = [1, 10]
        tested_covariance_types = ['spherical', 'diag', 'full']
        tested_mutations = ['?']
        tested_recombinations = ['?']
        tested_popsizes = ['?']
        if args.bigstates:
            tested_max_iters = [50]
        else:   
            tested_max_iters = [50, 100, 150]
    elif args.method == 'fcm 1c':
        tested_max_iters = [150]
        tested_nos_random_initializations = ['?']
        tested_covariance_types = ['?']
        tested_mutations = [0.5]
        tested_recombinations = [0.5]
        tested_popsizes = [10]
    elif args.method == 'hmm 1c':
        tested_max_iters = [50]
        tested_nos_random_initializations = [1, 10]
        tested_covariance_types = ['spherical', 'diag', 'full']
        tested_mutations = ['?']
        tested_recombinations = ['?']
        tested_popsizes = ['?']
    elif args.method == 'vsfcm nn':
        tested_max_iters = [150]
        tested_nos_random_initializations = ['?']
        tested_covariance_types = ['?']
        tested_mutations = [0.5]
        tested_recombinations = [0.5]
        tested_popsizes = [10]

    parameters = list(itertools.product(
        tested_methods,
        tested_nums_states,
        tested_max_iters,
        tested_nos_random_initializations,
        tested_covariance_types,
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

        for method_name, no_states, max_iter, no_rand_init, cov_type, mut, recomb, popsize in parameters:
            for f in chosen_folds:
                was_already_tested = False
                for at_row in already_tested_rows:
                    if at_row[0] == str(dataset_name) and \
                        at_row[1] == str(method_name) and \
                        at_row[2] == str(f) and \
                        at_row[4] == str(no_states) and \
                        at_row[5] == str(max_iter) and \
                        at_row[13] == str(no_rand_init) and \
                        at_row[14] == str(cov_type) and \
                        at_row[15] == str(mut) and \
                        at_row[16] == str(recomb) and \
                        at_row[17] == str(popsize):
                        print(f"Skipping {dataset_name} {method_name, no_states, max_iter, no_rand_init, cov_type, mut, recomb, popsize} {f}")
                        was_already_tested = True
                        break
                if was_already_tested:
                    continue
                fold_train_xses_series = folds[f][0]
                fold_train_ys = folds[f][1]
                fold_validation_xses_series = folds[f][2]
                fold_validation_ys = folds[f][3]
                test_solution(
                    method_name,
                    fold_validation_xses_series,
                    fold_validation_ys,
                    fold_train_xses_series,
                    fold_train_ys,
                    dataset_name,
                    no_states,
                    max_iter,
                    f,
                    "",
                    csv_results_path,
                    no_random_initializations=no_rand_init,
                    covariance_type=cov_type,
                    mutation=mut,
                    recombination=recomb,
                    popsize=popsize)
