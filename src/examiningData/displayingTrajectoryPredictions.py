import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def draw_trajectory_with_predictions(xs, model, savepath):
    predictions = model.predict(xs)
    fig, ax = plt.subplots()
    plot_xs = [x[0] for x in xs]
    plot_ys = [x[1] for x in xs]
    arrow_width = 0.005
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.scatter(
        plot_xs,
        plot_ys,
        color='blue')
    for i in range(len(predictions)):
        ax.arrow(
            plot_xs[i],
            plot_ys[i],
            predictions[i][0] - plot_xs[i],
            predictions[i][1] - plot_ys[i],
            color='green',
            length_includes_head = True,
            width=arrow_width
        )

    ax.grid()
    plt.savefig(savepath)
    plt.close()


def draw_trajectory_predictions_gif(xs, model, savedir, gif_filename):
    unique_prefix = int(np.random.random()*10000)
    filenames = []
    for i in range(len(xs)-1):
        filename = savedir / f"{unique_prefix}_step_{i}.png"
        draw_trajectory_with_predictions(
            [xs[i], xs[i+1]],
            model,
            filename
        )
        filenames.append(filename)

    with imageio.get_writer(str(savedir / f"{gif_filename}.gif"), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)