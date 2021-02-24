import matplotlib.pyplot as plt


def display_series_with_different_markers(xses_series, plot_path, main_title, sub_titles, markerss):
    if len(xses_series[0][0]) != 2:
        print("Unable to visualize data with dimension other than 2")
        return

    width = 5
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