import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from matplotlib.ticker import LinearLocator, StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D
from loadingData import univariateDatasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    parser.add_argument('--states', required=False, action='store_true')
    parser.add_argument('--scatter', required=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    should_draw_scatter = args.scatter
    should_y_states = args.states

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    df = df[df['no_states'] <= '7']

    method_to_num_experiments = {}
    method_to_num_experiments['fcm nn'] = 360
    method_to_num_experiments['hmm nn'] = 270

    method_to_color = {}
    method_to_color['hmm nn'] = 'hotpink'
    method_to_color['fcm nn'] = 'royalblue'
    method_to_color['vsfcm nn'] = 'lightsteelblue'

    COLOR_COVARIANCE = True
    covariance_to_color = {}
    covariance_to_color['spherical'] = "pink"
    covariance_to_color['diag'] = "orchid"
    covariance_to_color['full'] = "hotpink"

    failed_color = 'grey'

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

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()


    for method, covariance in methods_and_covariances:
        method_df = df[df['method'] == method]
        method_df = method_df[method_df['covariance_type'] == covariance]
        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100, subplot_kw={"projection": "3d"})
        X = []
        Y = []
        Z = []
        for dataset in datasets:
            dataset_df = method_df[method_df['dataset'] == dataset]
            dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
            train_size = dataset_info[0]
            series_length = dataset_info[2]

            if method == 'hmm nn':
                chosen_params = hmm_chosen_params
            elif method == 'fcm nn':
                chosen_params = fcm_chosen_params
            else:
                chosen_params = vsfcm_chosen_params

            for key, value in chosen_params.items():
                dataset_df = dataset_df[dataset_df[key] == value]

            num_rows = dataset_df.shape[0]
            if num_rows != 3*5:
                print(f"Skipping {dataset} (only {num_rows} rows for {method})")
                continue

            y_to_mccs = {}

            for index, row in dataset_df.iterrows():
                no_states = float(row['no_states'])
                if should_y_states:
                    y=no_states
                else:
                    if method == 'hmm nn':
                        num_parameters = no_states*no_states+no_states
                        if covariance == 'spherical':
                            num_parameters += no_states
                        elif covariance == 'diag':
                            num_parameters += 2*no_states
                        elif covariance == 'full':
                            num_parameters += 4*no_states
                        else:
                            raise Exception("Unknown covariance type")
                    elif method == 'fcm nn':
                        num_parameters = no_states*no_states
                    else:
                        num_parameters = no_states*(no_states-1)
                    y=num_parameters
                
                if row['mcc'] == '?':
                    mcc=0
                else:
                    mcc=float(row['mcc'])
                if y in y_to_mccs.keys():
                    y_to_mccs[y].append(mcc)
                else:
                    y_to_mccs[y] = [mcc]

            dictitems = y_to_mccs.items()
            ys = []
            zs = []
            if should_draw_scatter:
                X_len = 15
                for y, mccs in dictitems:
                    for mcc in mccs:
                        ys.append(y)
                        zs.append(mcc)
            else:
                X_len = 5
                for y, mccs in dictitems:
                    ys.append(y)
                    z = sum(mccs)/len(mccs)
                    zs.append(z)
            X.extend(np.ones(X_len)*np.log(train_size*series_length))
            Y.extend(ys)
            Z.extend(zs)

        surf = ax.scatter(X, Y, Z, zdir='z', color=method_to_color[method])
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.02f}'))

        # color = method_to_color[method]
        # if method == 'hmm nn':
        #     color = covariance_to_color[covariance]

        if should_y_states:
            ax.set_ylabel('number of states/centers')
        else:
            ax.set_ylabel('number of trainable parameters')
        ax.set_xlabel('log(train_size * series_length)')
        ax.set_zlabel('mcc')

        if method == 'hmm nn':
            plt.suptitle(f'Performance of chosen {method} {covariance}')
        else:
            plt.suptitle(f'Performance of chosen {method}')
        plt.show()
        # plt.savefig(plots_dir / f'{dataset}.png')
        plt.close()
