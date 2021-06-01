import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    parser.add_argument('--numdatasets', '-n', required=False, default=30, type=int)
    args = parser.parse_args()
    return args


def render_plot(df, datasets, method):
    df = df[df['method'] == method]
    method_to_num_experiments = {}
    method_to_num_experiments['fcm nn'] = 360

    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)

    tested_datasets = []
    degenerated_shares = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        if dataset_df.shape[0] != method_to_num_experiments[method]:
            print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows for {method})")
            continue

        tested_datasets.append(dataset)
        degenerated_shares.append(np.mean([float(ds) for ds in dataset_df['degenerated_share']]))


    tested_datasets = [td[0:10] for td in tested_datasets]
    ax.bar(tested_datasets, degenerated_shares, color='royalblue')

    ax.set_ylabel('mean degenerated weights\' share')
    ax.set_ylim([-0.05,1.05])
    plt.xticks(rotation=45)
    
    if method == 'fcm nn':
        title = f'{method}'
    else:
        title = f'{method}'
    plt.suptitle(title)
    # plt.show()
    plt.savefig(plots_dir / f'{title}.png')
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())
    df = df[df['no_states'] <= '7']

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    datasets = datasets[0:args.numdatasets]

    render_plot(df, datasets, "fcm nn")
