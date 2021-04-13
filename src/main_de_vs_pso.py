from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import pyswarms as ps
from scipy.optimize import differential_evolution

from loadingData import univariateDatasets
from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.psoCognitiveMap import PSOCognitiveMap
from loadingData import loadSktime
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing


def test_models_on_real_dataset(dataset_name, no_states):
    derivative_order = 1

    max_iter_range = list(range(1, 10))
    max_iter_range.extend(list(range(10, 252, 10)))

    data_loading_function=loadSktime.load_sktime
    test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts')
    train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts')

    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    centers, transformed_train_xses_series = cmeans.find_centers_and_transform(
        xses_series=train_xses_series,
        c=no_states)

    pso_errors = []
    de_errors = []

    
    start_timestamp = time.time()
    for max_iter in tqdm(max_iter_range):
        pso_model = PSOCognitiveMap(no_states)
        pso_model.train([transformed_train_xses_series[0]], max_iter)
        pso_error = pso_model.get_error(transformed_train_xses_series[0])
        pso_errors.append(pso_error)
    pso_time = time.time() - start_timestamp
    
    start_timestamp = time.time()
    for max_iter in tqdm(max_iter_range):
        de_model = DECognitiveMap(no_states)
        de_model.train([transformed_train_xses_series[0]], max_iter)
        de_error = de_model.get_error(transformed_train_xses_series[0])
        de_errors.append(de_error)
    de_time = time.time() - start_timestamp

    return list(max_iter_range), de_errors, pso_errors, de_time, pso_time


def _constant0dot5_for_training_pso(x):
    results = []
    for i in range(x.shape[0]): 
        results.append(np.sum(np.abs(x[i]-0.5)))
    return results


def _constant0dot5(x):
    return np.sum(np.abs(x-0.5))


def _cos10x_for_training_pso(x):
    results = []
    for i in range(x.shape[0]): 
        results.append(np.sum(np.cos(10*x[i])))
    return results


def _cos10x(x):
    return np.sum(np.cos(10*x))


def test_bare_methods_on_simple_function(no_dimensions, simple_function, simple_function_for_training_pso):

    max_iter_range = range(1, 1002, 5)
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    x_max = np.ones(no_dimensions)
    x_min = -1 * x_max
    bounds_pso = (x_min, x_max)
    bounds_de = [(-1, 1) for _ in range(no_dimensions)]

    pso_errors = []
    de_errors = []

    pso_time = 0.0
    de_time = 0.0

    for max_iter in max_iter_range:
        start_timestamp = time.time()
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10,
            dimensions=no_dimensions,
            options=options,
            bounds=bounds_pso)

        cost, pos = optimizer.optimize(
            simple_function_for_training_pso,
            iters=max_iter)
        
        pso_time += time.time() - start_timestamp
        
        pso_error = simple_function(pos)
        pso_errors.append(pso_error)

        start_timestamp = time.time()
        result = differential_evolution(
            simple_function,
            bounds_de,
            None,
            maxiter=max_iter,
            popsize=10,
            strategy='rand1bin',
            mutation=0.8,
            recombination=0.9,
            init='random',
            seed=1)
        
        de_time += time.time() - start_timestamp
        no_de_iterations = result.nit

        de_error = simple_function(result.x)
        de_errors.append(de_error)

    print(f"pso time: {pso_time}")
    print(f"de time: {de_time}")
    return list(max_iter_range), de_errors, pso_errors, de_time, pso_time, no_de_iterations


def display_results(xs, de_errors, pso_errors, title):
    fig, ax = plt.subplots()
    ax.plot(xs, pso_errors, label='PSO')
    ax.plot(xs, de_errors, label='DE')
    ax.legend()
    ax.set_title(title)
    plt.show()
    plt.close()



if __name__ == "__main__":
    datasets = univariateDatasets.DATASETS_NAMES_WITH_NUMBER_OF_CLASSES

    fig, axs = plt.subplots(2)
    de_times = []
    pso_times = []
    # de_iters = []

    dimensions_range = range(2, 11, 2)
    for no_dimensions in dimensions_range:
        # xs, de_errors, pso_errors, de_time, pso_time, de_iter = test_bare_methods_on_simple_function(no_dimensions, _constant0dot5, _constant0dot5_for_training_pso)
        xs, de_errors, pso_errors, de_time, pso_time = test_models_on_real_dataset(datasets[5][0], no_dimensions)
        axs[0].plot(xs, de_errors, label=f'n={no_dimensions}')
        axs[1].plot(xs, pso_errors, label=f'n={no_dimensions}')
        de_times.append(de_time)
        pso_times.append(pso_time)
        # de_iters.append(de_iter)

    axs[0].legend()
    axs[1].legend()
    axs[0].set_title(f"Differential evolution estimating {datasets[5][0]}")
    axs[1].set_title(f"PSO estimating {datasets[5][0]}")
    plt.show()
    plt.close()

    display_results(list(dimensions_range), de_times, pso_times, "Time of execution [s]")

    # fig, ax = plt.subplots()
    # ax.plot(dimensions_range, de_iters, label='DE')
    # ax.legend()
    # ax.set_title("Number of iterations performed")
    # plt.show()
    # plt.close()

    # no_dimensions = 2
    # xs, de_errors, pso_errors = test_bare_methods_on_simple_function(no_dimensions, _cos10x, _cos10x_for_training_pso)
    # display_results(xs, de_errors, pso_errors, f"cos(10x), dim(x)={no_dimensions}")



