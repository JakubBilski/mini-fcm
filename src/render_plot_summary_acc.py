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
    parser.add_argument('--numdatasets', '-n', required=False, type=int, default=30)
    parser.add_argument('--best', '-b', required=False, action='store_true')
    args = parser.parse_args()
    return args


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

    methods_and_covariances = []
    methods_and_covariances.append(("fcm nn", "?"))
    # methods_and_covariances.append(("vsfcm nn", "?"))
    methods_and_covariances.append(("hmm nn", "spherical"))
    methods_and_covariances.append(("hmm nn", "diag"))
    methods_and_covariances.append(("hmm nn", "full"))

    hmm_chosen_params = {}
    hmm_chosen_params['maxiter'] = '150'
    hmm_chosen_params['no_random_initializations'] = '10'

    fcm_chosen_params = {}
    fcm_chosen_params['maxiter'] = '150'
    fcm_chosen_params['mutation'] = '0.5'
    fcm_chosen_params['recombination'] = '0.5'
    fcm_chosen_params['popsize'] = '10'

    vsfcm_chosen_params = {}
    vsfcm_chosen_params['maxiter'] = '150'
    vsfcm_chosen_params['mutation'] = '0.5'
    vsfcm_chosen_params['recombination'] = '0.5'
    vsfcm_chosen_params['popsize'] = '10'

    method_to_color = {}
    method_to_color['hmm nn'] = 'hotpink'
    method_to_color['fcm nn'] = 'royalblue'
    method_to_color['vsfcm nn'] = 'lightsteelblue'

    covariance_to_color = {}
    covariance_to_color['spherical'] = "pink"
    covariance_to_color['diag'] = "orchid"
    covariance_to_color['full'] = "hotpink"

    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)

    for method, covariance in methods_and_covariances:
        method_df = df[df['method'] == method]
        method_df = df[df['covariance_type'] == covariance]
        if method == 'hmm nn':
            chosen_params = hmm_chosen_params
        elif method == 'fcm nn':
            chosen_params = fcm_chosen_params
        else:
            chosen_params = vsfcm_chosen_params

        for key, value in chosen_params.items():
            method_df = method_df[method_df[key] == value]

        tested_datasets = []
        mccs = []

        for dataset in datasets:
            dataset_df = method_df[method_df['dataset'] == dataset]
            if dataset_df.shape[0] != 15:
                print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows for {method})")
                continue

            if args.best:
                best_acc = -0.1
                for no_states in range(3, 8):
                    state_df = dataset_df[dataset_df['no_states'] == str(no_states)]
                    acc = np.mean([float(acc) if acc!='?' else 0.0 for acc in state_df['accuracy']])
                    if acc > best_acc:
                        best_acc = acc
                tested_datasets.append(dataset)
                mccs.append(best_acc)
            else:
                tested_datasets.append(dataset)
                mccs.append(np.mean([float(acc) if acc!='?' else 0.0 for acc in dataset_df['accuracy']]))

        tested_datasets = [td[0:6]+"..."+td[-1] if len(td) > 8 else td for td in tested_datasets]
        if method == 'hmm nn':
            label = method + " " + covariance
            color = covariance_to_color[covariance]
        else:
            label = method
            color = method_to_color[method]
        ax.plot(tested_datasets, mccs, color=color)
        ax.scatter(tested_datasets, mccs, color=color, label=label)

    ax.set_ylabel('accuracy')
    ax.set_ylim([-0.05,1.05])
    plt.xticks(rotation=45)
    if args.best:
        plt.suptitle('Accuracy for the best number of states')
    else:
        plt.suptitle('Mean accuracy for number of states/centers 3-7')
    # plt.show()
    plt.savefig(plots_dir / f'plot.png')
    plt.close()

