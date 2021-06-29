import argparse
import csv
import os

from loadingData import univariateDatasets


class bcolors:
    OKGREEN = '\033[92m'
    RED = '\033[41m'
    YELLOW = '\033[103m'
    END = '\x1b[0m'


NO_COLUMNS = 3


def parse_args():
    parser = argparse.ArgumentParser(description='Display exxperiments progress')
    parser.add_argument('--filename', '-f', required=True, type=str)
    args = parser.parse_args()
    return args


def display_problem_progress(rows, max_rows_for_dataset):    
    datasets = list(univariateDatasets.DATASET_NAME_TO_INFO.keys())
    longest_ds = max([len(x) for x in datasets])
    column_length = (len(datasets) - 1) // NO_COLUMNS + 1
    rows_to_print = [[] for _ in range(column_length)]
    for i in range(len(datasets)):
        dataset = datasets[i]
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
        rows_to_print[i % column_length].append(
            f"{dataset.ljust(longest_ds)} {bar_fill} {str(count).rjust(3)}/{max_rows_for_dataset}"
        )
    rows_to_print = ['  '.join(row) for row in rows_to_print]
    print("".join(["_" for _ in range(len(rows_to_print[0]))]))
    for row in rows_to_print:
        print(row)
    print()


if __name__ == '__main__':
    args = parse_args()
    csv_results_file = open(args.filename, 'r', newline='')
    csv_rows = list(csv.reader(csv_results_file))[1:]

    print("FMC: number of classes 3-7")
    print("maxiter [150]")
    print("mutation [0.5]")
    print("recombination [0.5]")
    print("popsize [10]")
    newrows = [row for row in csv_rows if row[1] == 'fcm nn' and int(row[4]) in [3, 4, 5, 6, 7] and int(row[5]) == 150 and float(row[15]) == 0.5 and float(row[16]) == 0.5 and int(row[17]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 15)

    print("FMC: number of classes 8, 9, 12, 16")
    print("maxiter [150]")
    print("mutation [0.5]")
    print("recombination [0.5]")
    print("popsize [10]")
    newrows = [row for row in csv_rows if row[1] == 'fcm nn' and int(row[4]) in [8, 9, 10, 12, 16] and int(row[5]) == 150 and float(row[15]) == 0.5 and float(row[16]) == 0.5 and int(row[17]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 15)

    print("FMC one per class: number of classes 3-7")
    print("maxiter 150")
    print("mutation 0.5")
    print("recombination 0.5")
    print("popsize 10")
    newrows = [row for row in csv_rows if row[1] == 'fcm 1c' and int(row[4]) in [3, 4, 5, 6, 7] and int(row[5]) == 150 and float(row[15]) == 0.5 and float(row[16]) == 0.5 and int(row[17]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 15)
        
    print("FMC one per class: number of classes 8, 9, 10, 12")
    print("maxiter 150")
    print("mutation 0.5")
    print("recombination 0.5")
    print("popsize 10")
    newrows = [row for row in csv_rows if row[1] == 'fcm 1c' and int(row[4]) in [8, 9, 10, 12, 16] and int(row[5]) == 150 and float(row[15]) == 0.5 and float(row[16]) == 0.5 and int(row[17]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 15)

    print("HMM: number of states 3-7")
    print("maxiter 50")
    print("num random inits 10")
    print("covariance type ['spherical', 'diag', 'full']")
    newrows = [row for row in csv_rows if row[1] == 'hmm nn' and int(row[4]) in [3, 4, 5, 6, 7] and int(row[5]) == 50 and int(row[13]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 45)

    print("HMM: number of states 8, 9, 10, 12")
    print("maxiter [50]")
    print("num random inits [1, 10]")
    print("covariance type ['spherical', 'diag', 'full']")
    newrows = [row for row in csv_rows if row[1] == 'hmm nn' and int(row[4]) in [8, 9, 10, 12, 16] and int(row[5]) == 50 and int(row[13]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 45)


    print("HMM one per class: number of states 3-7")
    print("maxiter [50]")
    print("num random inits 10")
    print("covariance type ['spherical', 'diag', 'full']")
    print(csv_rows[0])
    newrows = [row for row in csv_rows if row[1] == 'hmm 1c' and int(row[4]) in [3, 4, 5, 6, 7] and int(row[5]) == 50 and int(row[13]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 45)


    print("HMM one per class: number of states 8, 9, 10, 12, 16")
    print("maxiter [50]")
    print("num random inits 10")
    print("covariance type ['spherical', 'diag', 'full']")
    print(csv_rows[0])
    newrows = [row for row in csv_rows if row[1] == 'hmm 1c' and int(row[4]) in [8, 9, 10, 12, 16] and int(row[5]) == 50 and int(row[13]) == 10]
    if len(newrows) == 0:
        print("____________________\nNo experiments found\n")
    else:
        display_problem_progress(newrows, 45)


    # print("VSFMC: number of classes 3-7")
    # print("maxiter [150]")
    # print("mutation [0.5]")
    # print("recombination [0.5]")
    # print("popsize [10]")
    # rows = [row for row in csv_rows if row[1] == 'vsfcm nn' and int(row[4]) in [3, 4, 5, 6, 7]]
    # if len(rows) == 0:
    #     print("____________________\nNo experiments found\n")
    # else:
    #     display_problem_progress(rows, 15)
