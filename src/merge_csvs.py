import argparse
import csv
import os
from tqdm import tqdm

from savingResults import csvSettings

def parse_args():
    parser = argparse.ArgumentParser(description='Merge csvs into one. Remove duplicate expreriments.')
    parser.add_argument('--dirname', '-d', required=True, type=str)
    parser.add_argument('--resultname', '-n', required=False, type=str, default='merge_result.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gathered_rows = []
    for filename in os.listdir(args.dirname):
        if filename.endswith(".csv"):
            print(f"Found file: {filename}")
            csv_results_file = open(os.path.join(args.dirname, filename), 'r', newline='')
            csv_rows = list(csv.reader(csv_results_file))[1:]
            gathered_rows.extend([row for row in csv_rows])
    unique_gathered_rows = []
    meaningful_columns = csvSettings.get_indexes_characterizing_experiments()
    for row in tqdm(gathered_rows):
        duplicate = False
        for urow in unique_gathered_rows:
            duplicate = True
            for i in meaningful_columns:
                if row[i] != urow[i]:
                    duplicate = False
                    break
            if duplicate:
                break
        if not duplicate:
            unique_gathered_rows.append(row)

    print(f"Found {len(unique_gathered_rows)} unique rows")
    unique_datasets = list(set(row[0] for row in unique_gathered_rows))
    for dataset in unique_datasets:
        count = 0
        for row in unique_gathered_rows:
            if row[0] == dataset:
                count += 1
        print(f"{dataset}: {count} rows")

    csv_results_file = open(os.path.join(args.dirname, args.resultname), 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(csvSettings.get_header())
    csv_writer.writerows(unique_gathered_rows)
    csv_results_file.close()
