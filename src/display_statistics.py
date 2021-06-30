import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
from scipy.stats import spearmanr
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    args = parser.parse_args()
    return args


def biggest_difference_in_de(df):
    df = df[df['no_states'].astype(int) <= 7]
    df = df[df['method'] == 'fcm nn']
    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    biggest_difference = 0.0
    mean_difference = 0.0
    bd_dataset = ""
    for num_states in ['3', '4', '5', '6', '7']:
        state_df = df[df['no_states'] == num_states]
        for fold in ['0', '1', '2']:
            fold_df = state_df[state_df['fold_no'] == fold]
            for dataset in datasets:
                dataset_df = fold_df[fold_df['dataset'] == dataset]
                accuracies = dataset_df['accuracy'].astype(float)
                if len(accuracies) == 0:
                    print(f"Skipping {dataset} states {num_states} fold {fold}")
                    continue
                difference = max(accuracies) - min(accuracies)
                if difference > biggest_difference:
                    biggest_difference = difference
                    bd_dataset = dataset
                mean_difference += difference
    mean_difference /= (len(datasets) * 3 * 5) 
    print(f"Biggest difference in df: {biggest_difference} for {bd_dataset}")
    print(f"Mean difference in df: {mean_difference}")


def stats_in_bw(df):
    df = df[df['no_states'].astype(int) <= 7]
    df = df[df['method'] == 'hmm nn']

    for covariance in ['full', 'spherical', 'diag']:
        cov_df = df[df['covariance_type'] == covariance]
        mean_acc_max_iter_50 = cov_df[cov_df['maxiter'] == '50']['accuracy'].replace('?', '0.0').astype(float).mean()
        mean_acc_max_iter_100 = cov_df[cov_df['maxiter'] == '100']['accuracy'].replace('?', '0.0').astype(float).mean()
        mean_acc_max_iter_150 = cov_df[cov_df['maxiter'] == '150']['accuracy'].replace('?', '0.0').astype(float).mean()
        print(f"covariance {covariance}")
        print(f"Mean accuracy when maxiter 50: {mean_acc_max_iter_50}")
        print(f"Mean accuracy when maxiter 100: {mean_acc_max_iter_100}")
        print(f"Mean accuracy when maxiter 150: {mean_acc_max_iter_150}")

        mean_acc_random_inits_1 = cov_df[cov_df['no_random_initializations'] == '1']['accuracy'].replace('?', '0.0').astype(float).mean()
        mean_acc_random_inits_10 = cov_df[cov_df['no_random_initializations'] == '10']['accuracy'].replace('?', '0.0').astype(float).mean()
        print(f"mean_acc_random_inits_1: {mean_acc_random_inits_1}")
        print(f"mean_acc_random_inits_10: {mean_acc_random_inits_10}")


def best_hyperparameters(df, expected_num_experiments, method_name):
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    for dataset_index in range(len(datasets)):
        dataset = datasets[dataset_index]
        dataset_df = df[df['dataset'] == dataset]
        if len(dataset_df) != expected_num_experiments:
            print(f"Only {len(dataset_df)} rows found for {dataset}")
            continue
        parameter_pairs_to_acc = {}
        # print()
        # print(f"{dataset}")
        for index, row in dataset_df.iterrows():
            parameter_pair = (row.no_states, row.covariance_type)
            accuracy = float(row.accuracy) if row.accuracy != "?" else 0.0
            if parameter_pair in parameter_pairs_to_acc.keys():
                parameter_pairs_to_acc[parameter_pair] += accuracy
            else:
                parameter_pairs_to_acc[parameter_pair] = accuracy
        best_accuracy = 0.0
        best_no_states = None
        best_covariance = None
        for k, v in parameter_pairs_to_acc.items():
            # print(f"{k}: {v}")
            if v > best_accuracy:
                best_accuracy = v
                best_no_states = k[0]
                best_covariance = k[1]
        # print(f"{dataset_index}: {best_no_states}, {best_covariance}")
        dirname = "test_" + method_name.replace(" ", "") + str(dataset_index)
        print(f'-d {dataset_index} -m {method_name} -rd {dirname} --test --teststates {best_no_states} --testcov {best_covariance}')


def correlations_between_methods(df):
    methods_and_covs = [
        ('fcm nn', '?'),
        ('fcm 1c', '?'),
        ('hmm nn', 'spherical'),
        ('hmm nn', 'diag'),
        ('hmm nn', 'full'),
        ('hmm 1c', 'spherical'),
        ('hmm 1c', 'diag'),
        ('hmm 1c', 'full')]
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    method_index_to_accs = []
    for method, cov in methods_and_covs:
        accs = []
        method_df = df[df['method'] == method]
        method_df = method_df[method_df['covariance_type'] == cov]
        for dataset_index in range(len(datasets)):
            dataset = datasets[dataset_index]
            dataset_df = method_df[method_df['dataset'] == dataset]
            mean_acc = dataset_df['accuracy'].replace('?', '0.0').astype(float).mean()
            accs.append(mean_acc)
        method_index_to_accs.append(accs)

    coefficients = []
    for i in range(len(methods_and_covs)):
        coefficients_i = []
        for j in range(len(methods_and_covs)):
            print(f"{methods_and_covs[i]} and {methods_and_covs[j]}")
            print(spearmanr(method_index_to_accs[i], method_index_to_accs[j])[0])
            coefficients_i.append(spearmanr(method_index_to_accs[i], method_index_to_accs[j])[0])
        coefficients.append(coefficients_i.copy())
        coefficients_i = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(np.asarray(coefficients), cmap='hot', interpolation='nearest', vmin=0.5, vmax=1.0)
    for i in range(len(methods_and_covs)):
        for j in range(len(methods_and_covs)):
            text = ax.text(j, i, f"{coefficients[i][j]:.2f}",
                        ha="center", va="center", color="black")
    ax.set_xticks(range(len(methods_and_covs)))
    ax.set_yticks(range(len(methods_and_covs)))
    ax.set_xticklabels([a.upper() + " " + (b[:4] if b != "?" else "") for a, b in methods_and_covs])
    ax.set_yticklabels([a.upper() + " " + (b[:4] if b != "?" else "") for a, b in methods_and_covs])
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.suptitle('Values of spearman coefficient between accuracy in datasets')
    plt.show()

def fcmnn_df(df):
    mdf = df[df['method'] == 'fcm nn']
    mdf = mdf[mdf['maxiter'].astype(int) == 150]
    mdf = mdf[mdf['mutation'].astype(float) == 0.5]
    mdf = mdf[mdf['recombination'].astype(float) == 0.5]
    mdf = mdf[mdf['popsize'].astype(int) == 10]
    return mdf


def fcm1c_df(df):
    mdf = df[df['method'] == 'fcm 1c']
    mdf = mdf[mdf['maxiter'].astype(int) == 150]
    mdf = mdf[mdf['mutation'].astype(float) == 0.5]
    mdf = mdf[mdf['recombination'].astype(float) == 0.5]
    mdf = mdf[mdf['popsize'].astype(int) == 10]
    return mdf


def hmmnn_df(df):
    mdf = df[df['method'] == 'hmm nn']
    mdf = mdf[mdf['maxiter'].astype(int) == 50]
    mdf = mdf[mdf['no_random_initializations'].astype(int) == 10]
    return mdf


def hmm1c_df(df):
    mdf = df[df['method'] == 'hmm 1c']
    mdf = mdf[mdf['maxiter'].astype(int) == 50]
    mdf = mdf[mdf['no_random_initializations'].astype(int) == 10]
    return mdf


if __name__ == "__main__":
    args = parse_args()

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    # biggest_difference_in_de(df)
    # stats_in_bw(df)

    # df = df[df['no_states'].astype(int) <= 12]

    # # print('Best hyperparameters for fcm nn')
    # best_hyperparameters(fcmnn_df(df), 30, 'fcm nn')
    # # # print('Best hyperparameters for fcm 1c')
    # best_hyperparameters(fcm1c_df(df), 30, 'fcm 1c')
    # # # print('Best hyperparameters for hmm nn')
    # best_hyperparameters(hmmnn_df(df), 10*3*3, 'hmm nn')
    # # # print('Best hyperparameters for hmm 1c')
    best_hyperparameters(hmm1c_df(df), 10*3*3, 'hmm 1c')

    # correlations_between_methods(df)
