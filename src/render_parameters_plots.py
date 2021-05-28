import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from loadingData import univariateDatasets
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Create plot 1')
    parser.add_argument('--filepath', '-f', required=True, type=str)
    parser.add_argument('--plotdir', '-d', required=False, type=str, default=f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    parser.add_argument('--scatter', required=False, action='store_true')
    parser.add_argument('--states', required=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plots_dir = Path(args.plotdir)
    os.mkdir(plots_dir)

    should_draw_scatter = args.scatter
    should_x_states = args.states

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

    COLOR_COVARIANCE = True
    covariance_to_color = {}
    covariance_to_color['spherical'] = "pink"
    covariance_to_color['diag'] = "orchid"
    covariance_to_color['full'] = "hotpink"

    methods_and_covariances = []
    methods_and_covariances.append(("fcm nn", "?"))
    methods_and_covariances.append(("hmm nn", "spherical"))
    methods_and_covariances.append(("hmm nn", "diag"))
    methods_and_covariances.append(("hmm nn", "full"))

    datasets = list(set(df['dataset']))
    datasets = [d for d in datasets if d in list(univariateDatasets.DATASET_NAME_TO_INFO.keys())]
    datasets.sort()

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        skip = False
        for method, num_experiments in method_to_num_experiments.items():
            num_rows = dataset_df[dataset_df['method'] == method].shape[0]
            if num_rows != method_to_num_experiments[method]:
                print(f"Skipping {dataset} (only {num_rows} rows for {method})")
                skip = True
                break
        if skip:
            continue

        fig, ax = plt.subplots(1, figsize=(8, 8), dpi=100)
        for method, covariance in methods_and_covariances:
            method_df = dataset_df[dataset_df['method'] == method]
            covariance_df = method_df[method_df['covariance_type'] == covariance]
            xs = []
            accuracies = []

            for index, row in covariance_df.iterrows():
                no_states = float(row['no_states'])
                accuracy = float(row['accuracy']) if row['accuracy']!='?' else 0.0
                accuracies.append(accuracy)
                if should_x_states:
                    xs.append(no_states)
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
                        raise Exception("Unknown method")
                    xs.append(num_parameters)

            if not should_draw_scatter:
                dictionary = {}
                for x in xs:
                    dictionary[x] = []
                for x, acc in zip(xs, accuracies):
                    dictionary[x].append(acc)
                dictitems = dictionary.items()
                xs = [x for x, accs in dictitems]
                accuracies = [sum(accs)/len(accs) for x, accs in dictitems]

            color = method_to_color[method]
            label = method
            if method == 'hmm nn':
                color = covariance_to_color[covariance]
                label += f" {covariance}"
            if should_draw_scatter:
                ax.scatter(xs, accuracies, color=color, label=label)
            else:
                ax.scatter(xs, accuracies, color=color, label=label)
                ax.plot(xs, accuracies, color=color)

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
            ax.set_ylabel('mean accuracy')
        ax.set_ylim([-0.05,1.05])
        plt.legend()
        plt.suptitle(f'{dataset} ({no_classes} classes, train size {train_size}, series len {series_length})')
        # plt.show()
        plt.savefig(plots_dir / f'{dataset}.png')
        plt.close()
