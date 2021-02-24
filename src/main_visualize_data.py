import os
from tqdm import tqdm
import pathlib
import numpy as np
from datetime import datetime

from examiningData import displaying
from transformingData import cmeans
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadArff
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def visualize_different_centers_in_data(load_function, test_path, train_path, no_classes, no_centers, no_trajectories, displayed_data_name):
    test_xses_series, test_ys = load_function(test_path)
    train_xses_series, train_ys = load_function(train_path)
    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)


    clustered_centers, _ = cmeans.find_centers_and_transform(train_xses_series, no_centers)
    no_random_centers = 100
    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)
    centerss = [np.multiply(np.asarray([np.random.rand(2) for _1 in range(no_centers)]), maxs-mins)+mins
        for _ in range(no_random_centers)]
    centerss.append(clustered_centers)

    scores = []

    for centers in tqdm(centerss):
        train_xses_series_transformed = cmeans.transform(
            xses_series=train_xses_series,
            centers=centers)
        train_models = []

        for xs, y in zip(train_xses_series_transformed, train_ys):
            mppi = MppiCognitiveMap(no_centers)
            mppi.train(xs)
            mppi.set_class(y)
            train_models.append(mppi)
    
        test_xses_series_transformed = cmeans.transform(
            xses_series=test_xses_series,
            centers=centers)

        test_models = []

        for xs, y in zip(test_xses_series_transformed, test_ys):
            mppi = MppiCognitiveMap(no_centers)
            mppi.train(xs)
            mppi.set_class(y)
            test_models.append(mppi)

        rf_accuracy = accuracyComparing.get_accuracy(
            train_models=train_models,
            test_models=test_models,
            test_xs=test_xses_series_transformed,
            input_size=no_centers,
            no_classes=no_classes)

        scores.append(rf_accuracy)


    buff = sorted(zip(scores[:-1], centerss[:-1]), key=lambda z: z[0])
    sub_titles = [f'{s:.2f}' for s, c in buff]
    new_centerss = [c for s, c in buff]

    take_only_best = min(9, no_random_centers)
    sub_titles = sub_titles[-take_only_best:]
    new_centerss = new_centerss[-take_only_best:]

    sub_titles.append(f'clusters {scores[-1]:.2f}')
    new_centerss.append(centerss[-1])

    plot_title = f"{displayed_data_name} different centers"
    displaying.display_series_with_different_markers(
        train_xses_series,
        plots_dir / f"{plot_title}.png",
        plot_title,
        sub_titles=sub_titles,
        markerss=new_centerss)

    plot_title = f"{displayed_data_name} first {no_trajectories} trajectories"
    displaying.display_trajectories_with_different_markers(
        train_xses_series,
        plots_dir / f"{plot_title}.png",
        plot_title,
        sub_titles=sub_titles,
        markerss=new_centerss,
        no_trajectories=no_trajectories)



if __name__ == "__main__":
    os.mkdir(plots_dir)

    visualize_different_centers_in_data(
        loadArff.load_atrial_fibrilation,
        pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TEST.arff'),
        pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TRAIN.arff'),
        3,
        10,
        5,
        "AtrialFibrillation")

    visualize_different_centers_in_data(
        loadArff.load_libras,
        pathlib.Path('./data/Libras/Libras_TEST.arff'),
        pathlib.Path('./data/Libras/Libras_TRAIN.arff'),
        15,
        10,
        10,
        "Libras")