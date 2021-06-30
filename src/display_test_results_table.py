import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
from scipy.stats import spearmanr
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--sotapath', '-s', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    csv_results_file = open(args.sotapath, 'r', newline='')
    csv_rows = list(csv.reader(csv_results_file))[1:]

    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    sota_algorithms = [
        "TS-CHIEF",
        "HIVE-COTE v1.0",
        "ROCKET",
        "InceptionTime",
        "STC",
        "ResNet",
        "ProximityForest",
        "WEASEL",
        "S-BOSS",
        "cBOSS",
        "BOSS",
        "RISE",
        "TSF",
        "Catch22"]

    for row in csv_rows:
        dataset = row[0]
        if dataset not in univariateDatasets.DATASET_NAME_TO_INFO.keys():
            continue
        sota_max = max(row[1:])
        sota_best = sota_algorithms[row.index(sota_max)-1]
        sota_max = f"{float(max(row[1:]))*100 : .2f}"
        dataset_df = df[df['dataset'] == dataset]
        # print(dataset_df[dataset_df['method'] == 'fcm nn']['accuracy'])

        mdf = dataset_df[dataset_df['method'] == 'fcm nn']
        fcmnn_acc = f"{float(mdf['accuracy'])*100 : .2f}"
        fcmnn_states = int(mdf['no_states'])

        mdf = dataset_df[dataset_df['method'] == 'fcm 1c']
        fcm1c_acc = f"{float(mdf['accuracy'])*100 : .2f}"
        fcm1c_states = int(mdf['no_states'])

        mdf = dataset_df[dataset_df['method'] == 'hmm nn']
        hmmnn_acc = f"{float(mdf['accuracy'])*100 : .2f}"
        hmmnn_states = int(mdf['no_states'])
        hmmnn_cov = mdf['covariance_type'].iloc(0)[0][:4]

        mdf = dataset_df[dataset_df['method'] == 'hmm 1c']
        hmm1c_acc = f"{float(mdf['accuracy'])*100 : .2f}"
        hmm1c_states = int(mdf['no_states'])
        hmm1c_cov = mdf['covariance_type'].iloc(0)[0][:4]

        # dataset_sota_df = sota_df[sota_df['TESTACC'] == dataset].drop(axis=1, labels='TESTACC')

        dataset_short = dataset[:8]

        # print(f"{dataset_short}&{sota_max}&{sota_best}&{fcmnn_acc}&{fcmnn_states}&{fcm1c_acc}&{fcm1c_states}&{hmmnn_acc}&{hmmnn_states} {hmmnn_cov}&{hmm1c_acc}&{hmm1c_states} {hmm1c_cov}\\\\\\hline")
        print(f"{dataset_short}&{sota_max}&{sota_best}&{hmmnn_acc}&{hmmnn_states} {hmmnn_cov}&{hmm1c_acc}&{hmm1c_states} {hmm1c_cov}&{fcmnn_acc}&{fcmnn_states}&{fcm1c_acc}&{fcm1c_states}\\\\\\hline")

