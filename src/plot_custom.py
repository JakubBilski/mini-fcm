import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from datetime import datetime
from loadingData import univariateDatasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    method_to_x_keys = {}
    method_to_x_keys['hmm nn'] = ['no_states', 'maxiter', 'no_random_initializations', 'covariance_type']
    method_to_x_keys['fcm nn'] = ['no_states', 'maxiter', 'mutation', 'recombination', 'popsize']

    method_to_num_experiments = {}
    method_to_num_experiments['fcm nn'] = 360
    method_to_num_experiments['hmm nn'] = 270

    df = df[df['no_states'] <= '7']

    for method, x_keys in method_to_x_keys.items():
        method_df = df[df['method'] == method]
        datasets = list(set(method_df['dataset']))
        datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
        datasets.sort()

        for dataset in datasets:
            dataset_df = method_df[method_df['dataset'] == dataset]

            if dataset_df.shape[0] != method_to_num_experiments[method]:
                print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows)")
                continue

            fig, axs = plt.subplots(1, len(x_keys), figsize=(16, 5), dpi=100)
            accuracies = [float(a) if a!="?" else 0.0 for a in dataset_df['accuracy']]

            for k in range(len(x_keys)):
                x_key = x_keys[k]
                axs[k].scatter(dataset_df[x_key], accuracies)
                distinct_xs_means = []
                distinct_xs = sorted(list(set(dataset_df[x_key])))
                axs[k].set_xticks(distinct_xs)
                for dx in distinct_xs:
                    dx_accuracies = dataset_df[dataset_df[x_key] == dx]['accuracy']
                    mean_dx_accuracy = np.mean([float(a) if a!="?" else 0.0 for a in dx_accuracies])
                    distinct_xs_means.append(f"{mean_dx_accuracy:.2f}")

                axs[k].set_xlabel(x_key)
                axs[k].margins(x=1/len(distinct_xs))
                twinaxis = axs[k].twiny()
                twinaxis.set_xticks(axs[k].get_xticks())
                # make ticks invisible, but keep labels
                twinaxis.tick_params(length=0, colors='blue')
                twinaxis.set_xticklabels(distinct_xs_means)
                twinaxis.set_xbound(axs[k].get_xbound())
                if k==0:
                    axs[k].set_ylabel('accuracy')
                else:
                    axs[k].set_yticklabels([])

            dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
            no_classes = dataset_info[3]
            train_size = dataset_info[0]
            series_length = dataset_info[2]
            plt.suptitle(f'{method} {dataset} ({no_classes} classes, train size {train_size}, series len {series_length})')
            # plt.show()
            plt.savefig(plots_dir / f'{method}_{dataset}.png')
            plt.close()



    