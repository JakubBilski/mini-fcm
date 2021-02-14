# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

from cognitiveMaps import mppiCheckpoints
from cognitiveMaps import basicComparing
from cognitiveMaps import rfClassifier
from cognitiveMaps import svmClassifier
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from transformingData import cmeans
from transformingData import derivatives
from loadingData import loadArff
from datetime import datetime

TEST_ONLY_RF = True

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def compare_solutions(train_models, test_models, test_xs, input_size, no_classes):
    mistakes = 0
    if not TEST_ONLY_RF:
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit, best_cost = basicComparing.nn_weights(train_models_without_same, test_model, input_size, input_size)
            if best_fit.get_class() != test_model.get_class():
                good_predictions = [m for m in train_models_without_same if m.get_class()==test_model.get_class()]
                best_correct_fit, best_correct_cost = basicComparing.nn_weights(good_predictions, test_model, input_size, input_size)
                mistakes += 1
            acc = 1-mistakes/len(test_models)
            print(f"nn_weights accuracy: {len(test_models)-mistakes}/{len(test_models)} ({acc})")


    if not TEST_ONLY_RF:
        if test_xs:
            mistakes = 0
            for test_model, xs in zip(test_models, test_xs):
                train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
                best_fit = basicComparing.nn_convergence(train_models_without_same, test_model, xs[0])
                if best_fit.get_class() != test_model.get_class():
                    mistakes += 1
            print(f"nn_convergence accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")


    if not TEST_ONLY_RF:
        mistakes = 0
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            fit_class, _ = basicComparing.best_mse_sum(train_models_without_same, test_model, no_classes)
            if fit_class != test_model.get_class():
                mistakes += 1
        print(f"best_mse_sum accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")


    mistakes = 0
    cfr = rfClassifier.RFClassifier(
        [tm.weights for tm in train_models],
        [tm.class_name for tm in train_models]
        )
    for test_model in test_models:
        fit_class = cfr.predict(test_model.weights)
        if fit_class != test_model.get_class():
            mistakes += 1
    print(f"rfClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    rf_accuracy = 1-mistakes/len(test_models)

    if not TEST_ONLY_RF:
        mistakes = 0
        cfr = svmClassifier.SVMClassifier(
            [tm.weights for tm in train_models],
            [tm.class_name for tm in train_models]
            )
        for test_model in test_models:
            fit_class = cfr.predict(test_model.weights)
            if fit_class != test_model.get_class():
                mistakes += 1
        print(f"SVMClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    return rf_accuracy


def generate_mppi_checkpoints(derivative_order, no_centers):
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/mppi/{no_centers}_{derivative_order}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    centers, xses_series = cmeans.find_centers_and_transform(xses_series, c=no_centers)

    mppiCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        input_size=no_centers,
        cmeans_centers=centers)

    input_path = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/mppi/{no_centers}_{derivative_order}/test/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    xses_series = cmeans.transform(xses_series, centers=centers)

    mppiCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        input_size=no_centers,
        cmeans_centers=centers)


def test_different_cmeans(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title):
    np.random.seed = 0

    no_random_centers = 25

    for no_centers in range(3, 20):
        random_centerss = [[np.random.rand(input_size) for _1 in range(no_centers)] for _ in range(no_random_centers)]
        random_centerss.append(cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers)[0])
        plot_xs, plot_ys = [], []
        for step in tqdm(range(no_random_centers+1)):
            random_centers = random_centerss[step]
            train_xses_series_transformed = cmeans.transform(
                xses_series=train_xses_series,
                centers=random_centers)
            
            train_models = []

            for xs, y in zip(train_xses_series_transformed, train_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                train_models.append(mppi)
        
            test_xses_series_transformed = cmeans.transform(
                xses_series=test_xses_series,
                centers=random_centers)

            test_models = []

            for xs, y in zip(test_xses_series_transformed, test_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                test_models.append(mppi)

            rf_accuracy = compare_solutions(
                train_models=train_models,
                test_models=test_models,
                test_xs=test_xses_series_transformed,
                input_size=no_centers,
                no_classes=no_classes)

            err = sum([mppi.get_error(xs) for mppi, xs in zip(train_models, test_xses_series_transformed)])/len(train_models)
            print(f'Prediction error: {err}')
            plot_xs.append(err)
            plot_ys.append(rf_accuracy)

        fig, ax = plt.subplots()
        ax.plot(plot_xs[:-1], plot_ys[:-1], 'bo')
        ax.set(xlabel='prediction error', ylabel='classification accuracy (rf)',
        title=f'{plot_title}, no_centers {no_centers}')
        ax.grid()

        plt.scatter(plot_xs[no_random_centers], plot_ys[no_random_centers], color='red')
        plt.savefig(plots_dir / f'{plot_title} no_centers {no_centers}.png')
        plt.close()


if __name__ == "__main__":

    os.mkdir(plots_dir)

    # solution comparison for many derivative cmeans mppi
    test_input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    test_xses_series, test_ys = loadArff.load_cricket_normalized(test_input_file)
    
    train_input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    train_xses_series, train_ys = loadArff.load_cricket_normalized(train_input_path)

    input_size = 6
    no_classes = 12
    plot_title = 'Cricket random centers vs cmeans'
    
    test_different_cmeans(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title)

    test_input_file = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')
    test_xses_series, test_ys = loadArff.load_uwave_normalized(test_input_file)
    
    train_input_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff')
    train_xses_series, train_ys = loadArff.load_uwave_normalized(train_input_path)

    input_size = 3
    no_classes = 8
    plot_title = 'Uwave random centers vs cmeans'
    
    test_different_cmeans(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title)