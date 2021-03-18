import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

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

def compare_fcm_hmm(csv_dir):

    hmm_xs_by_dataset = {}
    hmm_ys_by_dataset = {}
    fcm_xs_by_dataset = {}
    fcm_ys_by_dataset = {}

    no_classes_by_dataset = {}

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            csv_file = open(csv_dir / file, newline='')
            reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            lines = [line[0].split(sep=',') for line in reader]
            dataset_name = lines[1][0]
            no_classes = lines[1][1]
            no_classes_by_dataset[dataset_name] = no_classes

            if file.endswith("fcm_csv_path.csv"):
                fcm_xs_by_dataset[dataset_name] = [row[1] for row in lines[3:]]
                fcm_ys_by_dataset[dataset_name] = [float(row[2]) for row in lines[3:]]
            elif file.endswith("hmm_csv_path.csv"):
                hmm_xs_by_dataset[dataset_name] = [row[0] for row in lines[3:]]
                hmm_ys_by_dataset[dataset_name] = [float(row[1]) for row in lines[3:]]
            else:
                raise Exception("Unknown csv file suffix")
    
    for dataset_name in no_classes_by_dataset.keys():
        fig, ax = plt.subplots()
        ax.plot(
            fcm_xs_by_dataset[dataset_name],
            fcm_ys_by_dataset[dataset_name],
            color='blue',
            label=f'fcm, m = 2')
        ax.plot(
            hmm_xs_by_dataset[dataset_name],
            hmm_ys_by_dataset[dataset_name],
            color='red',
            label=f'hmm')
        ax.set(
            xlabel='# centers / # states',
            ylabel='classification accuracy',
            title=f'{dataset_name} ({no_classes_by_dataset[dataset_name]} classes) fcm vs hmm classification')
        ax.grid()
        ax.legend()
        plt.savefig(csv_dir / f'hmm vs fcm {dataset_name}.png')
        plt.close()

            


if __name__ == "__main__":
    csv_dir = pathlib.Path('D:\\Projekty\\fcm\\mini-fcm\\plots\\start')
    compare_fcm_hmm(csv_dir)
