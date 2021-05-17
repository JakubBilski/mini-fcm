import argparse
import csv
import os

from loadingData import univariateDatasets

class bcolors:
    OKGREEN = '\033[92m'
    RED = '\033[41m'
    YELLOW = '\033[103m'
    END = '\x1b[0m'

from savingResults import csvSettings

def parse_args():
    parser = argparse.ArgumentParser(description='Merge csvs into one. Remove duplicate expreriments.')
    parser.add_argument('--filename', '-f', required=True, type=str)
    args = parser.parse_args()
    return args


def display_problem_progress(rows, max_rows_for_dataset):    
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    longest_ds = max([len(x) for x in datasets])
    for dataset in datasets:
        count = 0
        for row in rows:
            if row[0] == dataset:
                count += 1
        if count == max_rows_for_dataset:
            bar_fill = "|\033[102m      \x1b[0m|"
        elif count > max_rows_for_dataset/2:
            bar_fill = "|\033[103m    \x1b[0m  |"
        elif count > 0:
            bar_fill = "|\033[41m  \x1b[0m    |"
        else:
            bar_fill = "|      |"
        print(f"{dataset.ljust(longest_ds)} {bar_fill} {count}/{max_rows_for_dataset}")

if __name__ == '__main__':
    args = parse_args()
    csv_results_file = open(args.filename, 'r', newline='')
    csv_rows = list(csv.reader(csv_results_file))[1:]

    print("FMC: number of classes 3-7")
    print("_______________________________")
    rows = [row for row in csv_rows if row[1] == 'fcm nn' and int(row[4]) in [3, 4, 5, 6, 7]]
    display_problem_progress(rows, 360)
    print("_______________________________")

    print("FMC: number of classes 8-9, 12, 16")
    print("_______________________________")
    rows = [row for row in csv_rows if row[1] == 'fcm nn' and int(row[4]) in [8, 9, 12, 16]]
    display_problem_progress(rows, 288)
    print("_______________________________")
