import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

from examiningData.displaying import display_comparison

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def compare_ms(csv_dir):
    
    ms = ['1.25', '1.5', '2.0', '4.0', '8.0', '16.0']
    colors = {ms[0]: "orange", ms[1]: "purple", ms[2]: "red", ms[3]: "blue", ms[4]: "green", ms[5]: "brown"}

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            csv_file = open(csv_dir / file, newline='')
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            lines = [line[0].split(sep=',') for line in reader]
            dataset_name = lines[1][0]
            no_classes = lines[1][1]
            plots_xs = {}
            plots_ys = {}
            for m in ms:
                plots_xs[m] = []
                plots_ys[m] = []

            fig, ax = plt.subplots()

            for row in lines[3:]:
                m = row[0]
                no_centers = row[1]
                accuracy = row[2]
                plots_xs[m].append(no_centers)
                plots_ys[m].append(float(accuracy))

            for m in ms:
                ax.plot(plots_xs[m], plots_ys[m], color=colors[m], label=f'm = {m}')

            ax.set(xlabel='# centers', ylabel='classification accuracy', title=f'{dataset_name} fcm classification')
            ax.grid()
            ax.legend()
            plt.savefig(csv_dir / f'{dataset_name} m comparison.png')
            plt.close()

          
def compare_classification_results(csv_dirs, results_dir):
    xs_by_dataset_by_method = {}
    ys_by_dataset_by_method = {}

    no_classes_by_dataset = {}

    for csv_dir in csv_dirs:
        for file in os.listdir(csv_dir):
            if file.endswith(".csv"):
                csv_file = open(csv_dir / file, newline='')
                reader = csv.reader(csv_file, delimiter='\n', quotechar='|')
                lines = [line[0].split(sep=',') for line in reader]
                dataset_name = lines[1][0]
                no_classes = lines[1][1]
                method_name = lines[1][2]
                no_classes_by_dataset[dataset_name] = no_classes
                if dataset_name not in xs_by_dataset_by_method.keys():
                    xs_by_dataset_by_method[dataset_name] = {}
                    ys_by_dataset_by_method[dataset_name] = {}
                xs_by_dataset_by_method[dataset_name][method_name] = [row[0] for row in lines[3:]]
                ys_by_dataset_by_method[dataset_name][method_name] = [float(row[1]) for row in lines[3:]]

    for dataset_name in no_classes_by_dataset.keys():
        display_comparison(
            f"{dataset_name} ({no_classes_by_dataset[dataset_name]} classes) classification",
            x_title='# centers / # states',
            y_title='classification accuracy',
            save_path=results_dir / f'{dataset_name} classification.png',
            plots_xs=list(xs_by_dataset_by_method[dataset_name].values()),
            plots_ys=list(ys_by_dataset_by_method[dataset_name].values()),
            labels=list(xs_by_dataset_by_method[dataset_name].keys())
        )


def compare_degeneration_results(csv_dirs, results_dir):
    xs_by_dataset_by_method = {}
    ys_by_dataset_by_method = {}

    no_classes_by_dataset = {}

    for csv_dir in csv_dirs:
        for file in os.listdir(csv_dir):
            if file.endswith(".csv"):
                csv_file = open(csv_dir / file, newline='')
                reader = csv.reader(csv_file, delimiter='\n', quotechar='|')
                lines = [line[0].split(sep=',') for line in reader]
                dataset_name = lines[1][0]
                no_classes = lines[1][1]
                method_name = lines[1][2]
                no_classes_by_dataset[dataset_name] = no_classes
                if dataset_name not in xs_by_dataset_by_method.keys():
                    xs_by_dataset_by_method[dataset_name] = {}
                    ys_by_dataset_by_method[dataset_name] = {}
                xs_by_dataset_by_method[dataset_name][method_name] = [row[0] for row in lines[3:]]
                ys_by_dataset_by_method[dataset_name][method_name] = [float(row[2]) for row in lines[3:]]

    for dataset_name in no_classes_by_dataset.keys():
        display_comparison(
            f"{dataset_name} ({no_classes_by_dataset[dataset_name]} classes) degeneration",
            x_title='# centers / # states',
            y_title='degenerated weights share',
            save_path=results_dir / f'{dataset_name} degeneration.png',
            plots_xs=list(xs_by_dataset_by_method[dataset_name].values()),
            plots_ys=list(ys_by_dataset_by_method[dataset_name].values()),
            labels=list(xs_by_dataset_by_method[dataset_name].keys())
        )

def compare_different_Ls(csv_dir, output_dir, title):
    mean_accuracy = {}
    for L in ["1", "1.1", "1.2", "1.3", "1.4", "1.5"]:
        mean_accuracy[L] = 0

    no_datasets = 0
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            no_datasets += 1
            csv_file = open(csv_dir / file, newline='')
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            lines = [line[0].split(sep=',') for line in reader]
            for row in lines[3:]:
                mean_accuracy[row[1]] += float(row[2])
    
    for k, v in mean_accuracy.items():
        mean_accuracy[k] = v/no_datasets

    fig, ax = plt.subplots()
    plot_xs = list(mean_accuracy.keys())
    plot_ys = list(mean_accuracy.values())
    ax.plot(
        plot_xs,
        plot_ys,
        color='blue',
        label=f'fcm, m = 2')
    ax.set(
        xlabel='L used in sigmoid',
        ylabel='mean classification accuracy',
        title=f'{title}')
    ax.grid()
    ax.legend()
    plt.savefig(output_dir / f'{title}.png')
    plt.close()



if __name__ == "__main__":
    # csv_dir = pathlib.Path('plots\\picked\sigmoid_L_examination_5_centers\\csvs')
    # output_dir = pathlib.Path('plots\\picked\sigmoid_L_examination_5_centers\\computed')
    # title = 'mean classification accuracy with no_centers 5'
    # compare_different_Ls(csv_dir, output_dir, title)

    os.mkdir(plots_dir)

    hmm_dir = pathlib.Path('plots\\picked\\hmm_classification_results')
    fcm_dir = pathlib.Path('plots\\picked\\tau_5_decm_tests\\csvs')
    # decm1c_dir = pathlib.Path('plots\\picked\\decm_1c')
    # hmm1c_dir = pathlib.Path('plots\\picked\\hmm_1c')
    oneit_dir = pathlib.Path('plots\\picked\\big_comparison_1it')

    csv_dirs = [fcm_dir, hmm_dir, oneit_dir]
    
    # all_dir = pathlib.Path('D:\\Projekty\\fcm\\mini-fcm\\plots\\picked\\arrowhead_tau_examination')
    # csv_dirs = [all_dir]

    compare_classification_results(csv_dirs, plots_dir)
    # compare_degeneration_results(csv_dirs, plots_dir)
