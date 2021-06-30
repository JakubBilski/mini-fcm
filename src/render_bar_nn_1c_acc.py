import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os

from display_statistics import fcm1c_df, fcmnn_df, hmm1c_df, hmmnn_df
from loadingData import univariateDatasets


def nn_vs_1c_barplot(df):
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())

    mdf = fcmnn_df(df)
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].astype(float).mean())
    fcmnn_acc = sum(accuracies)/len(accuracies)

    mdf = fcm1c_df(df)
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].astype(float).mean())
    fcm1c_acc = sum(accuracies)/len(accuracies)

    mdf = hmmnn_df(df)
    mdf = mdf[mdf['covariance_type'] == 'spherical']
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].replace('?', '0.0').astype(float).mean())
    hmmnnsph_acc = sum(accuracies)/len(accuracies)

    mdf = hmm1c_df(df)
    mdf = mdf[mdf['covariance_type'] == 'spherical']
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].replace('?', '0.0').astype(float).mean())
    hmm1csph_acc = sum(accuracies)/len(accuracies)

    mdf = hmmnn_df(df)
    mdf = mdf[mdf['covariance_type'] == 'diag']
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].replace('?', '0.0').astype(float).mean())
    hmmnndiag_acc = sum(accuracies)/len(accuracies)

    mdf = hmm1c_df(df)
    mdf = mdf[mdf['covariance_type'] == 'diag']
    accuracies = []
    for dataset in datasets:
        accuracies.append(mdf[mdf['dataset'] == dataset]['accuracy'].replace('?', '0.0').astype(float).mean())
    hmm1cdiag_acc = sum(accuracies)/len(accuracies)

    print("nn_vs_1c_barplot")
    print(fcmnn_acc)
    print(fcm1c_acc)
    print(hmmnnsph_acc)
    print(hmm1csph_acc)
    print(hmmnndiag_acc)
    print(hmm1cdiag_acc)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    x = np.arange(3)
    width = 0.35
    ax.bar(x-width/2, [fcmnn_acc, hmmnnsph_acc, hmmnndiag_acc], width, label='NN', color='mediumpurple')
    ax.bar(x+width/2, [fcm1c_acc, hmm1csph_acc, hmm1cdiag_acc], width, label='1C', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(['FCM', 'HMM spherical', 'HMM diagonal'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('mean accuracy', fontsize=13)
    plt.legend()
    fig.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)
    df = df[df['no_states'].astype(int) <= 16]

    nn_vs_1c_barplot(df)
    plt.savefig(plots_dir / "nn_vs_1c_acc.png", bbox_inches='tight')