import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

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

          
def compare_fcm_hmm(hmm_dir, fcm_dir, results_dir):

    hmm_xs_by_dataset = {}
    hmm_ys_by_dataset = {}
    fcm_xs_by_dataset = {}
    fcm_ys_by_dataset = {}

    no_classes_by_dataset = {}

    for file in os.listdir(hmm_dir):
        if file.endswith(".csv"):
            csv_file = open(hmm_dir / file, newline='')
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            lines = [line[0].split(sep=',') for line in reader]
            dataset_name = lines[1][0]
            no_classes = lines[1][1]
            no_classes_by_dataset[dataset_name] = no_classes


            hmm_xs_by_dataset[dataset_name] = [row[0] for row in lines[3:]]
            hmm_ys_by_dataset[dataset_name] = [float(row[1]) for row in lines[3:]]


    for file in os.listdir(fcm_dir):
        if file.endswith(".csv"):
            csv_file = open(fcm_dir / file, newline='')
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            lines = [line[0].split(sep=',') for line in reader]
            dataset_name = lines[1][0]
            no_classes = lines[1][1]
            no_classes_by_dataset[dataset_name] = no_classes
            fcm_xs_by_dataset[dataset_name] = [row[1] for row in lines[3:]]
            fcm_ys_by_dataset[dataset_name] = [float(row[2]) for row in lines[3:]]
    
    common_keys = fcm_xs_by_dataset.keys()
    common_keys = [ck for ck in common_keys if ck in hmm_xs_by_dataset.keys()]
    common_keys = [ck for ck in common_keys if ck in fcm_xs_by_dataset.keys()]

    for dataset_name in common_keys:
        fig, ax = plt.subplots()
        ax.plot(
            fcm_xs_by_dataset[dataset_name],
            fcm_ys_by_dataset[dataset_name],
            color='blue',
            label=f'fcm')
        ax.plot(
            hmm_xs_by_dataset[dataset_name],
            hmm_ys_by_dataset[dataset_name],
            color='red',
            label=f'hmm')
        ax.set(
            xlabel='# centers / # states',
            ylabel='classification accuracy',
            title=f'{dataset_name} ({no_classes_by_dataset[dataset_name]} classes) classification')
        ax.grid()
        ax.legend()
        plt.savefig(results_dir / f'{dataset_name} classification.png')
        plt.close()
  

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
    compare_fcm_hmm(hmm_dir, fcm_dir, plots_dir)
