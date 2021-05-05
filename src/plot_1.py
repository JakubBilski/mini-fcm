import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from datetime import datetime
from loadingData import univariateDatasets
import os


if __name__ == "__main__":
    plots_dir = Path(f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    os.mkdir(plots_dir)

    csv_path = Path('D:\\Projekty\\fcm\\mini-fcm\\plots\\picked\\final_data\\raw\\classification_results_2.csv')
    df = pd.read_csv(csv_path)
    print(df.head())

    method = 'hmm nn'
    df = df[df['method'] == method]

    datasets = list(set(df['dataset']))
    datasets.sort()

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

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



    