import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
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

if __name__ == "__main__":
    args = parse_args()

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    biggest_difference_in_de(df)
    stats_in_bw(df)
    