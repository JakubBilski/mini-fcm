import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def display_series(xses_series, plot_path, main_title):
    if len(xses_series[0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return
    height = 2
    width = 2
    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    plot_xs = [x[0] for xs in xses_series for x in xs]
    plot_ys = [x[1] for xs in xses_series for x in xs]
    axs[0,0].scatter(plot_xs, plot_ys, color='blue', s=1)
    axs[0,0].set(xlabel='x[0]', ylabel='x[1]', title=f'all data points')

    plot_xs = [x[0] for x in xses_series[0]]
    plot_ys = [x[1] for x in xses_series[0]]
    axs[0,1].plot(plot_xs, plot_ys, color='blue')
    mean_step = 0
    for i in range(len(plot_xs)-1):
        buff1 = (plot_xs[i+1] - plot_xs[i])
        buff2 = (plot_xs[i+1] - plot_xs[i])
        mean_step += np.sqrt(buff1*buff1+buff2*buff2)
    mean_step /= len(plot_xs)
    axs[0,1].set(
        xlabel='x[0]',
        ylabel='x[1]',
        title=f'first trajectory ({len(plot_xs)} points, mean step {mean_step:.2f})')

    colors = ['blue', 'red', 'green', 'orange', 'brown']
    color_index = 0
    for xs in xses_series[:5]:
        plot_xs = [x[0] for x in xs]
        plot_ys = [x[1] for x in xs]
        axs[1,0].plot(plot_xs, plot_ys, color=colors[color_index])
        color_index += 1
    axs[1,0].set(xlabel='x[0]', ylabel='x[1]', title=f'first 5 trajectories')

    plt.savefig(plot_path)
    plt.close()


def display_series_with_markers(xses_series, plot_path, main_title, markers):
    if len(xses_series[0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    marker_xs = [x[0] for x in markers]
    marker_ys = [x[1] for x in markers]

    height = 2
    width = 2
    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    plot_xs = [x[0] for xs in xses_series for x in xs]
    plot_ys = [x[1] for xs in xses_series for x in xs]
    axs[0,0].scatter(plot_xs, plot_ys, color='blue', s=1)
    axs[0,0].set(xlabel='x[0]', ylabel='x[1]', title=f'all data points')
    
    plot_xs = [x[0] for xs in xses_series for x in xs]
    plot_ys = [x[1] for xs in xses_series for x in xs]
    axs[0,1].scatter(plot_xs, plot_ys, color='blue', s=1)
    axs[0,1].scatter(marker_xs, marker_ys, color='red', s=10, zorder=10)
    axs[0,1].set(xlabel='x[0]', ylabel='x[1]', title=f'all data points with cmeans centers')

    plot_xs = [x[0] for x in xses_series[0]]
    plot_ys = [x[1] for x in xses_series[0]]
    axs[1,0].plot(plot_xs, plot_ys, color='blue')
    mean_step = 0
    for i in range(len(plot_xs)-1):
        buff1 = (plot_xs[i+1] - plot_xs[i])
        buff2 = (plot_xs[i+1] - plot_xs[i])
        mean_step += np.sqrt(buff1*buff1+buff2*buff2)
    mean_step /= len(plot_xs)
    axs[1,0].set(
        xlabel='x[0]',
        ylabel='x[1]',
        title=f'first trajectory ({len(plot_xs)} points, mean step {mean_step:.2f})')
    
    
    plot_xs = [x[0] for x in xses_series[0]]
    plot_ys = [x[1] for x in xses_series[0]]
    axs[1,1].plot(plot_xs, plot_ys, color='blue')
    axs[1,1].scatter(marker_xs, marker_ys, color='red', s=10, zorder=10)
    axs[1,1].set(
        xlabel='x[0]',
        ylabel='x[1]',
        title=f'first trajectory with centers')

    plt.savefig(plot_path)
    plt.close()


def display_series_with_different_markers(xses_series, plot_path, main_title, sub_titles, markerss):
    if len(xses_series[0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    width = max(min(len(sub_titles)//2, 5),1)

    height = (len(sub_titles) - 1) // width + 1
    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    plot_xs = [x[0] for xs in xses_series for x in xs]
    plot_ys = [x[1] for xs in xses_series for x in xs]
    a = 0
    b = 0
    for i in range(len(sub_titles)):
        axs[a,b].scatter(plot_xs, plot_ys, color='blue', s=1)
        plot_xs2 = [x[0] for x in markerss[i]]
        plot_ys2 = [x[1] for x in markerss[i]]
        axs[a,b].scatter(plot_xs2, plot_ys2, color='red')
        axs[a,b].set(xlabel='x[0]', ylabel='x[1]', title=f'{sub_titles[i]}')
        # axs[a,b].grid()
        b += 1
        if b == width:
            b = 0
            a += 1

    plt.savefig(plot_path)
    plt.close()


def display_different_series_with_different_markers(xses_seriess, plot_path, main_title, sub_titles, markerss):
    if len(xses_seriess[0][0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    sqr = int(np.sqrt(len(sub_titles)))
    if sqr*sqr == len(sub_titles):
        width = max(sqr,2)
        height = max(sqr,2)
    else:
        width = max(sqr+1,2)
        if sqr*(sqr+1) < len(sub_titles):
            height = max(sqr+1, 2)
        else:
            height = max(sqr, 2)
    xmax = 1
    xmin = 0
    ymax = 1
    ymin = 0

    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    a = 0
    b = 0
    for i in range(len(sub_titles)):
        plot_xs = [x[0] for xs in xses_seriess[i] for x in xs]
        plot_ys = [x[1] for xs in xses_seriess[i] for x in xs]
        plot_xs2 = [x[0] for x in markerss[i]]
        plot_ys2 = [x[1] for x in markerss[i]]
        axs[a,b].scatter(plot_xs, plot_ys, color='blue', s=1)
        axs[a,b].scatter(plot_xs2, plot_ys2, color='red', s=15)
        axs[a,b].set(xlabel='x[0]', ylabel='x[1]', title=f'{sub_titles[i]}')
        axs[a,b].set_xlim([xmin,xmax])
        axs[a,b].set_ylim([ymin,ymax])
        # axs[a,b].grid()
        b += 1
        if b == width:
            b = 0
            a += 1

    plt.savefig(plot_path)
    plt.close()


def display_hmm_and_cmeans_centers(xses_seriess, plot_path, main_title, sub_titles, centerss, covarss):
    if len(xses_seriess[0][0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    # covars must be diagonal for drawing to work

    sqr = int(np.sqrt(len(sub_titles)))
    if sqr*sqr == len(sub_titles):
        width = max(sqr,2)
        height = max(sqr,2)
    else:
        width = max(sqr+1,2)
        if sqr*(sqr+1) < len(sub_titles):
            height = max(sqr+1, 2)
        else:
            height = max(sqr, 2)
    xmax = 1
    xmin = 0
    ymax = 1
    ymin = 0
    ellipse_scale_in_std = 3

    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    a = 0
    b = 0
    for i in range(len(sub_titles)):
        plot_xs = [x[0] for xs in xses_seriess[i] for x in xs]
        plot_ys = [x[1] for xs in xses_seriess[i] for x in xs]
        plot_xs2 = [x[0] for x in centerss[i]]
        plot_ys2 = [x[1] for x in centerss[i]]

        axs[a,b].scatter(plot_xs, plot_ys, color='blue', s=1)
        axs[a,b].scatter(plot_xs2, plot_ys2, color='red', s=15)
        axs[a,b].set(xlabel='x[0]', ylabel='x[1]', title=f'{sub_titles[i]}')
        axs[a,b].set_xlim([xmin,xmax])
        axs[a,b].set_ylim([ymin,ymax])
        if i != len(sub_titles)-1:
            for j in range(len(centerss[i])):
                ellipse_width = np.sqrt(covarss[i][j][0][0])*ellipse_scale_in_std
                ellipse_height = np.sqrt(covarss[i][j][1][1])*ellipse_scale_in_std
                ellipse = Ellipse((centerss[i][j][0], centerss[i][j][1]),
                    width=ellipse_width, height=ellipse_height,
                    facecolor='pink', edgecolor='red', zorder=0)
                axs[a,b].add_patch(ellipse)
        b += 1
        if b == width:
            b = 0
            a += 1

    plt.savefig(plot_path)
    plt.close()

def display_trajectories_with_different_markers(xses_series, plot_path, main_title, sub_titles, markerss, no_trajectories):
    if len(xses_series[0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    width = 5
    height = (len(sub_titles) - 1) // width + 1
    fig, axs = plt.subplots(height, width)
    fig.set_size_inches(width*4.0, height*4.0)
    fig.tight_layout(pad=3.0)
    fig.suptitle(f'{main_title}')

    plot_xss = [[x[0] for x in xs] for xs in xses_series]
    plot_yss = [[x[1] for x in xs] for xs in xses_series]
    a = 0
    b = 0
    for i in range(len(sub_titles)):
        for t in range(min(no_trajectories, len(plot_xss))):
            axs[a,b].plot(plot_xss[t], plot_yss[t], color='blue', linewidth=1)
        plot_xs2 = [x[0] for x in markerss[i]]
        plot_ys2 = [x[1] for x in markerss[i]]
        axs[a,b].scatter(plot_xs2, plot_ys2, color='red', zorder=1)
        axs[a,b].set(xlabel='x[0]', ylabel='x[1]', title=f'{sub_titles[i]}')
        # axs[a,b].grid()
        b += 1
        if b == width:
            b = 0
            a += 1

    plt.savefig(plot_path)
    plt.close()


def display_comparison(title, x_title, y_title, save_path, plots_xs, plots_ys, labels):
    colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'black']
    no_plots = len(labels)
    if no_plots > len(colors):
        raise Exception(f"Number of plots ({no_plots}) cannot be greater than {len(colors)}")
    fig, ax = plt.subplots()
    for i in range(no_plots):
        ax.plot(
            plots_xs[i],
            plots_ys[i],
            color=colors[i],
            label=labels[i])

    ax.set(
        xlabel=x_title,
        ylabel=y_title,
        title=title)
    ax.grid()
    ax.legend()
    plt.savefig(save_path)
    plt.close()