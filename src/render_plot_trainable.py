import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    parser.add_argument('--scatter', required=False, action='store_true')
    parser.add_argument('--nofailedcircles', required=False, action='store_true')
    parser.add_argument('--states', required=False, action='store_true')
    parser.add_argument('--vsfcm', required=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    should_draw_scatter = args.scatter
    should_x_states = args.states
    should_failed_circles = not args.nofailedcircles
    with_vsfcm = args.vsfcm

    csv_path = args.filepath
    df = pd.read_csv(csv_path, dtype="str")
    print(df.head())

    df = df[df['no_states'] <= '7']

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
    if with_vsfcm:
        methods_and_covariances.append(("vsfcm nn", "?"))
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

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100)
        
        xs_one_failed = []
        ys_one_failed = []
        xs_two_failed = []
        ys_two_failed = []
        xs_three_failed = []
        ys_three_failed = []
        skip = False
        for method, covariance in methods_and_covariances:
            method_df = dataset_df[dataset_df['method'] == method]
            method_df = method_df[method_df['covariance_type'] == covariance]
            label = method
            if method == 'hmm nn':
                label += f" {covariance}"
                chosen_params = hmm_chosen_params
            elif method == 'fcm nn':
                chosen_params = fcm_chosen_params
            else:
                chosen_params = vsfcm_chosen_params

            for key, value in chosen_params.items():
                # label += f", {key}: {value}"
                method_df = method_df[method_df[key] == value]

            num_rows = method_df.shape[0]
            if num_rows != 3*5:
                print(f"Skipping {dataset} (only {num_rows} rows for {method})")
                skip = True
                break

            x_to_accuracies = {}
            x_to_nums_failed_learning = {}

            for index, row in method_df.iterrows():
                no_states = float(row['no_states'])
                if should_x_states:
                    x=no_states
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
                    x=num_parameters
                
                if row['accuracy'] == '?':
                    accuracy=0
                    num_failed_learning=1
                else:
                    accuracy=float(row['accuracy'])
                    num_failed_learning=0
                if x in x_to_accuracies.keys():
                    x_to_accuracies[x].append(accuracy)
                    x_to_nums_failed_learning[x] += num_failed_learning
                else:
                    x_to_accuracies[x] = [accuracy]
                    x_to_nums_failed_learning[x] = num_failed_learning

            dictitems = x_to_accuracies.items()
            xs = []
            ys = []
            if should_draw_scatter:
                for x, accs in dictitems:
                    for acc in accs:
                        xs.append(x)
                        ys.append(acc)
            else:
                for x, accs in dictitems:
                    xs.append(x)
                    y = sum(accs)/len(accs)
                    ys.append(y)
                    if x_to_nums_failed_learning[x] == 1:
                        xs_one_failed.append(x)
                        ys_one_failed.append(y)
                    elif x_to_nums_failed_learning[x] == 2:
                        xs_two_failed.append(x)
                        ys_two_failed.append(y)
                    elif x_to_nums_failed_learning[x] == 3:
                        xs_three_failed.append(x)
                        ys_three_failed.append(y)

            color = method_to_color[method]
            if method == 'hmm nn':
                color = covariance_to_color[covariance]
            if should_draw_scatter:
                ax.scatter(xs, ys, color=color, label=label)
            else:
                ax.scatter(xs, ys, color=color, label=label)
                ax.plot(xs, ys, color=color)

        if skip:
            plt.close()
            continue

        if should_failed_circles:
            marker_size = 3.0*mpl.rcParams['lines.markersize'] ** 2
            if len(xs_one_failed) > 0:
                ax.scatter(xs_one_failed, ys_one_failed, s=marker_size,
                    marker=mpl.markers.MarkerStyle('x'), linestyle="None",
                    edgecolors=failed_color, facecolors=failed_color, label='failed to learn one fold')
            if len(xs_two_failed) > 0:
                ax.scatter(xs_two_failed, ys_two_failed, s=marker_size,
                    marker=mpl.markers.MarkerStyle('d'), linestyle="None",
                    edgecolors=failed_color, facecolors=failed_color, label='failed to learn two folds')
            if len(xs_three_failed) > 0:
                ax.scatter(xs_three_failed, ys_three_failed, s=marker_size,
                    marker=mpl.markers.MarkerStyle('s', fillstyle='full'), linestyle="None",
                    edgecolors=failed_color, facecolors=failed_color, label='failed to learn three folds')

        dataset_info = univariateDatasets.DATASET_NAME_TO_INFO[dataset]
        no_classes = dataset_info[3]
        train_size = dataset_info[0]
        series_length = dataset_info[2]
        plt.subplots_adjust(wspace=0.0)
        if should_x_states:
            ax.set_xlabel('number of states/centers')
        else:
            ax.set_xlabel('number of trainable parameters')
        if should_draw_scatter:
            ax.set_ylabel('accuracy')
        else:
            ax.set_ylabel('mean 3-fold cross-validation accuracy')
        ax.set_ylim([-0.05,1.05])
        plt.legend()
        plt.suptitle(f'Performance of selected models: {dataset} ({no_classes} classes, train size {train_size}, series len {series_length})')
        # plt.show()
        plt.savefig(plots_dir / f'{dataset}.png')
        plt.close()
