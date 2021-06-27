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
    parser.add_argument('--fold_no', '-fn', required=False, type=int)
    args = parser.parse_args()
    return args


def appropriate_label(name, method):
    appropriate_labels = {}
    if method == 'hmm nn':
        appropriate_labels['no_states'] = "hidden states"
    else:
        appropriate_labels['no_states'] = "concepts"

    appropriate_labels['maxiter'] = "maximal iterations"
    appropriate_labels['no_random_initializations'] = "random initializations"
    appropriate_labels['covariance_type'] = "covariance type"
    if name in appropriate_labels.keys():
        return appropriate_labels[name]
    return name


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    df = df[df['no_states'].astype(int) <= 7]

    method_to_x_keys = {}
    method_to_x_keys['hmm nn'] = ['no_states', 'maxiter', 'no_random_initializations', 'covariance_type']
    method_to_x_keys['fcm nn'] = ['no_states', 'maxiter', 'mutation', 'recombination', 'popsize']

    method_to_num_experiments = {}
    method_to_num_experiments['fcm nn'] = 360
    method_to_num_experiments['hmm nn'] = 270

    if args.fold_no is not None:
        df = df[df['fold_no'].astype(int) == args.fold_no]
        for k in method_to_num_experiments.keys():
            method_to_num_experiments[k] /= 3

    method_to_color = {}
    method_to_color['hmm nn'] = 'hotpink'
    method_to_color['fcm nn'] = 'royalblue'

    for method, x_keys in method_to_x_keys.items():
        method_df = df[df['method'] == method]
        datasets = list(set(method_df['dataset']))
        datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
        datasets.sort()

        for dataset in datasets:
            dataset_df = method_df[method_df['dataset'] == dataset]

            if dataset_df.shape[0] != method_to_num_experiments[method]:
                print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows for {method})")
                continue

            fig, axs = plt.subplots(1, len(x_keys), figsize=(10, 4), dpi=100)

            for k in range(len(x_keys)):
                x_key = x_keys[k]
                distinct_xs = list(set(dataset_df[x_key]))

                # if labels are ints, sort them as ints
                try:
                    distinct_xs = [int(dx) for dx in distinct_xs]
                    distinct_xs.sort()
                    distinct_xs = [str(dx) for dx in distinct_xs]
                except ValueError:
                    pass

                to_violin = []
                for dx in distinct_xs:
                    dx_accuracies = dataset_df[dataset_df[x_key] == dx]['accuracy']
                    dx_accuracies = np.asarray([float(a) if a!="?" else 0.0 for a in dx_accuracies])
                    to_violin.append(np.asarray(dx_accuracies))
                ticks = range(len(distinct_xs))
                violin_parts = axs[k].violinplot(to_violin, ticks, showextrema=False, showmeans=True)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(method_to_color[method])
                    pc.set_alpha(1.0)
                violin_parts['cmeans'].set_linewidth(3.0)
                violin_parts['cmeans'].set_edgecolor("black")
                axs[k].set_xticks(ticks)
                axs[k].set_xticklabels(distinct_xs)
                axs[k].set_xlabel(appropriate_label(x_key, method))
                axs[k].set_ylim([-0.05,1.05])
                axs[k].margins(x=1/len(distinct_xs))
                axs[k].patch.set_alpha(0.0)
                if k==0:
                    axs[k].set_ylabel('accuracy')
                    axs[k].tick_params(labelleft=True, labelright=False,
                     left=True, right=True)
                elif k==len(x_keys)-1:
                    axs[k].tick_params(labelleft=False, labelright=True,
                     left=True, right=True)
                else:
                    axs[k].tick_params(labelleft=False, labelright=False,
                     left=True, right=True)

            dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
            no_classes = dataset_info[3]
            train_size = dataset_info[0]
            series_length = dataset_info[2]
            plt.subplots_adjust(wspace=0.0)
            # plt.suptitle(f'{method} {dataset} ({no_classes} classes, train size {train_size}, series len {series_length})')
            if args.fold_no is None:
                plt.suptitle(f'{dataset}')
            else:
                plt.suptitle(f'{dataset}, fold {args.fold_no}')
            # plt.show()
            plt.savefig(plots_dir / f'{method}_{dataset}.png', bbox_inches='tight')
            plt.close()



    