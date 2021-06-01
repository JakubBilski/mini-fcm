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

    method = 'hmm nn'
    df = df[df['method'] == method]

    datasets = list(set(df['dataset']))
    datasets.sort()

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        if dataset_df.shape[0] != 270:
            print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows)")
            continue

        cov_types = ['spherical', 'diag', 'full']
        fig, axs = plt.subplots(1, len(cov_types), figsize=(16, 5), dpi=100)

        for c in range(len(cov_types)):
            covariance_type = cov_types[c]
            cov_df = dataset_df[dataset_df['covariance_type'] == covariance_type]

            steps = sorted(list(set(cov_df['no_states'])), reverse=True)
            maxiters = sorted(list(set(cov_df['maxiter'])))
            no_inits = sorted(list(set(cov_df['no_random_initializations'])))
            bw_parameters = [(mi, noi) for noi in no_inits for mi in maxiters]

            all_accuracies = []
            for step in steps:
                step_df = cov_df[cov_df['no_states'] == step]
                step_accuracies = []
                for mi, noi in bw_parameters:
                    bw_params_df = step_df[step_df['maxiter'] == mi]
                    bw_params_df = bw_params_df[bw_params_df['no_random_initializations'] == noi]
                    accuracies = bw_params_df['accuracy']
                    if (accuracies == '?').any():
                        step_accuracies.append(0)
                    else:
                        accuracies = [float(a) for a in list(accuracies)]
                        step_accuracies.append(sum(accuracies)/len(accuracies))
                all_accuracies.append(step_accuracies)
        
            all_accuracies = np.asarray(all_accuracies)
            im = axs[c].imshow(np.asarray(all_accuracies), cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
            axs[c].set_xticks(np.arange(len(bw_parameters)))
            axs[c].set_yticks(np.arange(len(steps)))
            axs[c].set_xticklabels(bw_parameters, rotation = 45)
            axs[c].set_yticklabels(steps)
            axs[c].set_xlabel('Baum-Welch parameters')
            axs[c].set_ylabel('Hidden states')

            for i in range(len(steps)):
                for j in range(len(bw_parameters)):
                    text = axs[c].text(j, i, f"{all_accuracies[i, j]:.2f}",
                                ha="center", va="center", color="black")

            axs[c].set_title(f"{covariance_type}")

        cbar = fig.colorbar(im, ax=axs.tolist(), orientation="horizontal")
        cbar.ax.set_xlabel('Accuracy')
        dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
        no_classes = dataset_info[3]
        train_size = dataset_info[0]
        series_length = dataset_info[2]
        plt.suptitle(f'{dataset} ({no_classes} classes, {train_size}x{series_length} train)')
        # plt.show()
        plt.savefig(plots_dir / f'{dataset}.png')
        plt.close()



    