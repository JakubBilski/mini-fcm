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
    parser.add_argument('--numdatasets', '-n', required=False, type=int, default=85)
    parser.add_argument('--best', '-b', required=False, action='store_true')
    parser.add_argument('--vsfcm', required=False, action='store_true')
    parser.add_argument('--hmmnn', required=False, action='store_true')
    parser.add_argument('--hmm1c', required=False, action='store_true')
    parser.add_argument('--fcmnn', required=False, action='store_true')
    parser.add_argument('--fcm1c', required=False, action='store_true')
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

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    datasets = datasets[0:args.numdatasets]

    methods_and_covariances = []
    if args.vsfcm:
        methods_and_covariances.append(("vsfcm nn", "?"))
    if args.hmmnn:
        methods_and_covariances.append(("hmm nn", "spherical"))
    if args.hmm1c:
        methods_and_covariances.append(("hmm 1c", "spherical"))
    if args.fcmnn:
        methods_and_covariances.append(("fcm nn", "?"))
    # methods_and_covariances.append(("hmm nn", "diag"))
    # methods_and_covariances.append(("hmm nn", "full"))
    if args.fcm1c:
        methods_and_covariances.append(("fcm 1c", "?"))
    
    if len(methods_and_covariances) == 0:
        print("No methods selected, assuming default")
        methods_and_covariances.append(("hmm nn", "spherical"))
        methods_and_covariances.append(("hmm 1c", "spherical"))
        methods_and_covariances.append(("fcm 1c", "?"))
        methods_and_covariances.append(("fcm nn", "?"))


    hmm_chosen_params = {}
    hmm_chosen_params['maxiter'] = '50'
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
    method_to_color['hmm 1c'] = 'orange'
    method_to_color['fcm nn'] = 'royalblue'
    method_to_color['fcm 1c'] = 'forestgreen'
    method_to_color['vsfcm nn'] = 'grey'

    covariance_to_color = {}
    covariance_to_color['spherical'] = "pink"
    covariance_to_color['diag'] = "orchid"
    covariance_to_color['full'] = "hotpink"

    plot_data = []
    datasets_sum_accuracies = [0 for _ in range(len(datasets))]
    datasets_num_runs = [0 for _ in range(len(datasets))]

    for method, covariance in methods_and_covariances:
        method_df = df[df['method'] == method]
        method_df = method_df[method_df['covariance_type'] == covariance]
        if method == 'hmm nn' or method == 'hmm 1c':
            chosen_params = hmm_chosen_params
        elif method == 'fcm nn' or method == 'fcm 1c':
            chosen_params = fcm_chosen_params
        else:
            chosen_params = vsfcm_chosen_params

        for key, value in chosen_params.items():
            method_df = method_df[method_df[key] == value]

        tested_datasets = []
        mccs = []

        for i in range(len(datasets)):
            dataset_df = method_df[method_df['dataset'] == datasets[i]]
            if dataset_df.shape[0] != 15:
                print(f"Skipping {datasets[i]} (only {dataset_df.shape[0]} rows for {method})")
                continue

            if args.best:
                best_acc = -0.1
                for no_states in range(3, 8):
                    state_df = dataset_df[dataset_df['no_states'] == str(no_states)]
                    acc = np.mean([float(acc) if acc!='?' else 0.0 for acc in state_df['accuracy']])
                    if acc > best_acc:
                        best_acc = acc
                tested_datasets.append(i)
                mccs.append(best_acc)
            else:
                tested_datasets.append(i)
                mccs.append(np.mean([float(acc) if acc!='?' else 0.0 for acc in dataset_df['accuracy']]))

            datasets_sum_accuracies[i] += mccs[-1]
            datasets_num_runs[i] += 1

        if method == 'hmm nn' or method=='hmm 1c':
            label = method + " " + covariance
        else:
            label = method
        color = method_to_color[method]
        plot_data.append((tested_datasets, mccs, color, label))

    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)

    ticks_map = sorted([i for i in range(len(datasets))],
                       key=lambda x: datasets_sum_accuracies[x] / datasets_num_runs[x],
                       reverse=True)
    ax.set_xticks([i for i in range(len(datasets))])
    ticks_labels = [datasets[ticks_map[i]] for i in range(len(datasets))]
    ax.set_xticklabels([tl[0:6]+"..."+tl[-1] if len(tl) > 8 else tl for tl in ticks_labels])

    ax.set_ylabel('accuracy')
    ax.set_ylim([-0.05,1.05])

    for tested_datasets, mccs, color, label in plot_data:
        sorted_zip = sorted(zip(tested_datasets, mccs), key=lambda x: ticks_map.index(x[0]))
        # print(label)
        # for t, m in sorted_zip:
        #     print(f"{datasets[t]}: {m} ({ticks_map[t]})")
        tested_datasets = [ticks_map.index(td) for td, mcc in sorted_zip]
        mccs = [mcc for td, mcc in sorted_zip]
        ax.plot(tested_datasets, mccs, color=color)
        ax.scatter(tested_datasets, mccs, color=color, label=label)

    plt.xticks(rotation=80)
    plt.legend()
    plt.grid(which='major')
    if args.best:
        plt.suptitle('Accuracy for the best number of states/centers 3-7')
    else:
        plt.suptitle('Mean accuracy for number of states/centers 3-7')
    # plt.show()
    plt.savefig(plots_dir / f'plot.png')
    plt.close()

