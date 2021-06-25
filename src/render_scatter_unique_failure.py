import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os

from main import load_preprocessed_data


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

    should_x_states = False
    unique_point_decimal_places = 4
    method = 'hmm nn'
    covariance = 'full'
    chosen_params = {}
    chosen_params['maxiter'] = '50'
    chosen_params['no_random_initializations'] = '10'

    method_df = df[df['method'] == method]
    method_df = method_df[method_df['covariance_type'] == covariance]
    for key, value in chosen_params.items():
        method_df = method_df[method_df[key] == value]

    expected_no_rows = 15

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    plot_info = []

    for dataset in datasets[:30]:
        dataset_df = method_df[method_df['dataset'] == dataset]
        skip = False
        num_failed_learning_hmm_nn_full = 0

        num_rows = dataset_df.shape[0]
        if num_rows != expected_no_rows:
            print(f"Skipping {dataset} (only {num_rows} rows for {method})")
            skip = True
            break
            
        if skip:
            continue

        x_to_nums_failed_learning = {}

        for index, row in dataset_df.iterrows():
            no_states = float(row['no_states'])
            if should_x_states:
                x=no_states
            else:
                num_parameters = no_states*no_states+no_states
                num_parameters += 4*no_states
                x=num_parameters
            
            if row['accuracy'] == '?':
                num_failed_learning = 1
            else:
                num_failed_learning = 0
            if x in x_to_nums_failed_learning.keys():
                x_to_nums_failed_learning[x] += num_failed_learning
            else:
                x_to_nums_failed_learning[x] = num_failed_learning

        num_failed_learning_hmm_nn_full = sum(x_to_nums_failed_learning.values())            

        dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
        no_classes = dataset_info[3]
        train_size = dataset_info[0]
        series_length = dataset_info[2]
        
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            test_path=pathlib.Path('data', 'Univariate_ts', f'{dataset}', f'{dataset}_TEST.ts'),
            train_path=pathlib.Path('data', 'Univariate_ts', f'{dataset}', f'{dataset}_TRAIN.ts'),
            derivative_order=1)
        
        min_num_unique_points = series_length
        for xs in train_xses_series:
            num_unique_points = len(set(
                [(round(x[0], unique_point_decimal_places), round(x[1], unique_point_decimal_places)) for x in xs]
                ))
            if num_unique_points < min_num_unique_points:
                min_num_unique_points = num_unique_points

        plot_info.append((min_num_unique_points, num_failed_learning_hmm_nn_full))
        print(f'{dataset}: {min_num_unique_points} <= {series_length}')
        print(f'{num_failed_learning_hmm_nn_full} failed')

    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100)
    plt.subplots_adjust(wspace=0.0)
    ax.set_xlabel('minimal number of unique points in one series')
    ax.set_ylabel('num times learning was unsuccessful')
    ax.set_ylim([-0.1, 15.1])

    xs = [up for up, fl in plot_info]
    ys = [fl for up, fl in plot_info]
    ax.scatter(xs, ys)

    plt.suptitle(f'Number of unique points in series vs hmm nn full learning')
    # plt.show()
    plt.savefig(plots_dir / f'scatter_unique_failure.png')
    plt.close()
