import matplotlib.pyplot as plt
import pandas as pd
from pathlib import PurePosixPath
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from datetime import datetime
import os


def whiskers_plot(ax, x, dx, ys, color):
    ax.hlines(
        [y for y in ys],
        [x - dx for y in ys],
        [x + dx for y in ys],
        color=color)
    ax.vlines(
        [x],
        [min(ys)],
        [max(ys)],
        color=color)


if __name__ == "__main__":
    plots_dir = PurePosixPath(f'plots/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}/')
    os.mkdir(plots_dir)

    csv_path = PurePosixPath('plots/picked/pso_degeneration/classification_results.csv')
    df = pd.read_csv(csv_path)
    print(df.head())
    colors = {
        'pso nn': 'red',
        'fcm nn': 'blue',
        'vsfcm nn': 'orange'
    }

    whiskers_width = 0.08
    method_shift = 0.1

    datasets = list(set(df['dataset']))

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        methods = list(set(df['method']))
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=100)

        shift = - method_shift * (len(methods) - 1) / 2
        xticks = set()
        for method in methods:

            method_df = dataset_df[dataset_df['method'] == method]
            steps = list(set(method_df['no_states']))
            steps.sort()
            for step in steps:
                step_df = method_df[method_df['no_states'] == step]
                degenerated_shares = list(step_df['degenerated_share'])
                mean_no_iterations = list(step_df['mean_no_iterations'])
                execution_times = list(step_df['complete_execution_time'])
                accuracies = list(step_df['accuracy'])

                whiskers_plot(axs[0, 0], step + shift, whiskers_width, accuracies, colors[method])
                whiskers_plot(axs[0, 1], step + shift, whiskers_width, degenerated_shares, colors[method])
                whiskers_plot(axs[1, 0], step + shift, whiskers_width, mean_no_iterations, colors[method])
                whiskers_plot(axs[1, 1], step + shift, whiskers_width, execution_times, colors[method])
            
            shift += method_shift
            xticks = xticks.union(set(steps))

        fig.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in colors.items()], loc='center right')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].xaxis.set_major_locator(ticker.FixedLocator(list(xticks)))
        axs[0, 1].xaxis.set_major_locator(ticker.FixedLocator(list(xticks)))
        axs[1, 0].xaxis.set_major_locator(ticker.FixedLocator(list(xticks)))
        axs[1, 1].xaxis.set_major_locator(ticker.FixedLocator(list(xticks)))
        axs[0, 0].set_title('Accuracy')
        axs[0, 1].set_title('Degenerated weights share')
        axs[1, 0].set_title('Mean number of iterations')
        axs[1, 1].set_title('Execution time [s]')
        axs[1, 0].hlines(
            [200, 1000, 200],
            3, 8,
            colors = [colors['vsfcm nn'], colors['pso nn'], colors['fcm nn']],
            linestyles = 'dotted'
        )
        fig.suptitle(dataset)
        plt.savefig(plots_dir / f'{dataset}.png')
        plt.close()



    