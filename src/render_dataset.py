import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os

from main import load_preprocessed_data


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--plotdir', '-rd', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    parser.add_argument('--dataset_id', '-d', required=True, choices=range(0,85), type=int)
    parser.add_argument('--series_id', '-s', required=False, type=int)
    parser.add_argument('--arrows', '-a', required=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    dataset_name = datasets[args.dataset_id]
    num_classes = univariateDatasets.DATASET_NAME_TO_INFO[dataset_name][3]
    
    train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
        test_path=pathlib.Path('data', 'Univariate_ts', f'{dataset_name}', f'{dataset_name}_TEST.ts'),
        train_path=pathlib.Path('data', 'Univariate_ts', f'{dataset_name}', f'{dataset_name}_TRAIN.ts'),
        derivative_order=1)

    if args.series_id is not None:
        if args.series_id >= len(train_xses_series):
            raise argparse.ArgumentTypeError(f'--series_id cannot be greater than no. series ({len(train_xses_series)})')
        series_range = [args.series_id]
    else:
        series_range = range(len(train_xses_series))
    
    plots_infos = []

    for i in series_range:
        series = train_xses_series[i]
        xs1 = [x1 for x1, x2 in series]
        xs2 = [x2 for x1, x2 in series]
        y = train_ys[i]
        plots_infos.append((i, xs1, xs2, y))

    for i, xs1, xs2, y in plots_infos:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        plt.subplots_adjust(wspace=0.0)
        suptitle = f'{dataset_name}: series {i}'
        plt.suptitle(suptitle)
        ax.scatter(xs1, xs2, s=5, alpha=0.7)
        if args.arrows:
            ax.plot(xs1, xs2, linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        # plt.show()
        plt.savefig(plots_dir / f'{dataset_name}_{i}.png')
        plt.close()

    if args.series_id is None:
        for c in range(num_classes):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            plt.subplots_adjust(wspace=0.0)
            suptitle = f'{dataset_name}: class {c}'
            plt.suptitle(suptitle)
            for i, xs1, xs2, y in plots_infos:
                if y == c:
                    ax.scatter(xs1, xs2, s=1, alpha=0.5)
                    if args.arrows:
                        ax.plot(xs1, xs2, linewidth=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            plt.savefig(plots_dir / f'{dataset_name}_class{c}.png')
            plt.close()

    if args.series_id is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
        plt.subplots_adjust(wspace=0.0)
        suptitle = f'{dataset_name}'
        plt.suptitle(suptitle)
        for i, xs1, xs2, y in plots_infos:
            ax.scatter(xs1, xs2, s=1, alpha=0.5)
            if args.arrows:
                ax.plot(xs1, xs2, linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.savefig(plots_dir / f'{dataset_name}_all.png', bbox_inches='tight')
        plt.close()
