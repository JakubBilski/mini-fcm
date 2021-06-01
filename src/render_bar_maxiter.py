import matplotlib.pyplot as plt
import pandas as pd
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
    args = parser.parse_args()
    return args


def render_plot(df, method, covariance, num_states, num_inits, mutation, recombination, popsize):
    method_to_max_color = {}
    method_to_max_color['hmm nn'] = 'pink'
    method_to_max_color['fcm nn'] = 'cornflowerblue'

    method_to_mean_color = {}
    method_to_mean_color['hmm nn'] = 'hotpink'
    method_to_mean_color['fcm nn'] = 'royalblue'

    df = df[df['method'] == method]
    df = df[df['covariance_type'] == covariance]
    df = df[df['no_states'] == num_states]
    df = df[df['no_random_initializations'] == num_inits]
    df = df[df['mutation'] == mutation]
    df = df[df['recombination'] == recombination]
    df = df[df['popsize'] == popsize]

    if method == 'fcm nn':
        maxiter_thresholds = [150, 200, 250]
    elif method == 'hmm nn':
        maxiter_thresholds = [50, 100, 150]
    else:
        raise Exception(f"Unknown method {method}")

    df = df[df['maxiter'] == str(maxiter_thresholds[-1])]

    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)

    tested_datasets = []
    learning_failed_datasets = []
    means_iterations = []
    maxs_iterations = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        num_rows = dataset_df[dataset_df['method'] == method].shape[0]
        if num_rows != 3:
            print(f"Skipping {dataset} (only {num_rows} rows for {method})")
            continue
        tested_datasets.append(dataset)

        mean_iterations = 0
        max_iterations = 0
        for index, row in dataset_df.iterrows():
            if row['mean_no_iterations'] == '?':
                mean_iterations = maxiter_thresholds[-1]*3
                max_iterations = maxiter_thresholds[-1]
                learning_failed_datasets.append(dataset)
                break
            mean_iterations += float(row['mean_no_iterations'])
            max_iterations = max([max_iterations, float(row['max_no_iterations'])])
        mean_iterations /= 3
        means_iterations.append(mean_iterations)
        maxs_iterations.append(max_iterations)

    ax.bar(tested_datasets, maxs_iterations, color=method_to_max_color[method], label="max")
    ax.bar(tested_datasets, means_iterations, color=method_to_mean_color[method], label="mean")

    tested_datasets = [td[0:10] for td in tested_datasets]
    if method == 'hmm nn':
        failed_mask = [0 if d not in learning_failed_datasets else maxiter_thresholds[-1] for d in tested_datasets]
        ax.bar(tested_datasets, failed_mask, color='grey', label="learning failed")

    for thr in maxiter_thresholds:
        ax.axhline(thr, color='black',linestyle='--', linewidth=2)

    ax.set_ylabel('number of performed iterations')
    ax.set_ylim([0,maxiter_thresholds[-1]+10])
    plt.legend()
    plt.xticks(rotation=45)

    if method == 'fcm nn':
        title = f'{method} centers {num_states} mutation {mutation} recombination {recombination} popsize {popsize}'
    else:
        title = f'{method} states {num_states} covariance type {covariance} random inits {num_inits}'
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

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    datasets = datasets[0:20]

    nums_states = ['3', '4', '5', '6', '7']
    covariances = ['spherical', 'diag', 'full']
    nums_inits = ['1', '10']

    parameters = list(itertools.product(
        nums_states,
        covariances,
        nums_inits
    ))
    for num_states, covariance, num_inits in parameters:
        render_plot(df, "hmm nn", covariance, num_states, num_inits, '?', '?', '?')

    mutations = ['0.5', '0.8']
    recombinations = ['0.5', '0.9']
    popsizes = ['10', '15']

    parameters = list(itertools.product(
        nums_states,
        mutations,
        recombinations,
        popsizes
    ))
    for num_states, mutation, recombination, popsize in parameters:
        render_plot(df, "fcm nn", '?', num_states, '?', mutation, recombination, popsize)
