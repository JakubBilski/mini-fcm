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

    df = df[df['no_states'].astype(int) <= 7]

    num_experiments = 5*5
    color = 'royalblue'
    y_keys = ['degenerated_share', 'accuracy']

    method_df = df[df['method'] == 'fcm nn']
    datasets = list(set(method_df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    for dataset in datasets:
        dataset_df = method_df[method_df['dataset'] == dataset]

        if dataset_df.shape[0] != num_experiments:
            print(f"Skipping {dataset} (only {dataset_df.shape[0]} rows)")
            continue

        fig, axs = plt.subplots(2, 1, figsize=(7, 4), dpi=100)

        for k in range(len(y_keys)):
            y_key = y_keys[k]
            distinct_xs = list(set(dataset_df['additional_info']))
            # distinct_xs = [int(x[5:]) if x != "" else 5 for x in distinct_xs]
            # distinct_xs.sort()
            # distinct_xs = [str(dx) for dx in distinct_xs]

            to_violin = []
            for dx in distinct_xs:
                ys = dataset_df[dataset_df['additional_info'] == dx][y_key]
                ys = np.asarray([float(s) for s in ys])
                to_violin.append(np.asarray(ys))
            ticks = range(len(distinct_xs))

            distinct_xs = [int(x[5:]) if x != "" else 5 for x in distinct_xs]
            sort_zip = zip(distinct_xs, to_violin)
            sort_zip = sorted(sort_zip, key=lambda x: x[0])
            distinct_xs = [str(dx) for dx, ys in sort_zip]
            to_violin = [ys for dx, ys in sort_zip]

            violin_parts = axs[k].violinplot(to_violin, ticks, showextrema=False, showmeans=True)
            for pc in violin_parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(1.0)
            violin_parts['cmeans'].set_linewidth(3.0)
            violin_parts['cmeans'].set_edgecolor("black")
            axs[k].set_xticks(ticks)
            axs[k].set_xticklabels(distinct_xs)
            axs[k].set_ylim([-0.07,1.07])
            axs[k].margins(x=1/len(distinct_xs))
            axs[k].patch.set_alpha(0.0)
            if k==0:
                axs[k].set_ylabel('share of degenerated weights')
                axs[k].tick_params(labelbottom=False, labeltop=False,
                    bottom=True, top=True)
            else:
                axs[k].set_xlabel('tau')
                axs[k].set_ylabel('accuracy')
                axs[k].tick_params(labelbottom=True, labeltop=False,
                    bottom=True, top=True)

        dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
        no_classes = dataset_info[3]
        train_size = dataset_info[0]
        series_length = dataset_info[2]
        plt.subplots_adjust(hspace=0.0)
        # plt.suptitle(f'sigmoid tau: {dataset} ({no_classes} classes, train size {train_size}, series len {series_length})')
        plt.suptitle(f'{dataset}')
        # plt.show()
        plt.savefig(plots_dir / f'{dataset}.png', bbox_inches='tight')
        plt.close()

    