from tqdm import tqdm
from datetime import datetime
import os
# from multiprocessing.dummy import Pool as ThreadPool

import pathlib
from examiningConvergence.examineConvergence import examineConvergence

from loadingData import loadArff
from cognitiveMaps import cognitiveMap


train_path = pathlib.Path('./data/Cricket/Cricket_TRAIN.arff')
test_path = pathlib.Path('./data/Cricket/Cricket_TEST.arff')
# train_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff')
# test_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def weights_distance(weights_a, weights_b):
    result = 0
    for i in range(len(weights_a)):
        for j in range(len(weights_a[0])):
            result += abs(weights_a[i][j]-weights_b[i][j])
    return result


def nn_weights(models, m):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def nn_weights_and_start_values(models, m):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights)
        cost += sum(model.start_values-m.start_values)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def nn_convergence(models, m, first_input):
    best_cost = 100000
    best_model = None
    pnt = m.get_convergence_point(first_input)
    for model in models:
        m_pnt = model.get_convergence_point(first_input)
        cost = sum(pnt-m_pnt)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def test_fcm_nn():
    os.mkdir(plots_dir)
    input_size = 6
    models = []
    xses_series, ys = loadArff.load_cricket_normalized(train_path)
    for i in tqdm(range(0,len(ys))):
        fcm = cognitiveMap.FuzzyCognitiveMap(input_size)
        fcm.train(xses_series[i])
        fcm.set_class(ys[i])
        models.append(fcm)
        # fcm.display_plot(plots_dir / f"trained{i}.png")
        # fcm.display_plot()
        # examineConvergence(fcm)
    
    xses_series, ys = loadArff.load_cricket_normalized(test_path)
    mismatches = 0
    for i in tqdm(range(len(ys))):
        fcm = cognitiveMap.FuzzyCognitiveMap(input_size)
        fcm.train(xses_series[i])
        class_prediction = nn_weights(models, fcm).get_class()
        if class_prediction != ys[i]:
            mismatches += 1
            # fcm.display_plot(plots_dir / f"predicted{i}_mistake{class_prediction}_{ys[i]}.png")
        else:
            pass
            # fcm.display_plot(plots_dir / f"predicted{i}.png")
    
    print(f"Accuracy: {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")
    # 86% Cricket
    # 31% UWave

def append_ecm_to_models(data):
    xs, y, models, input_size, extend_size = data
    fcm = cognitiveMap.ExtendedCognitiveMap(input_size, input_size+extend_size)
    fcm.train(xs, steps=200)
    fcm.set_class(y)
    models.append(fcm)


def test_ecm_nn():
    os.mkdir(plots_dir)
    input_size = 6
    extend_size = 3
    models = []
    xses_series, ys = loadArff.load_cricket_normalized(train_path)
    # input_sizes = [input_size for i in range(len(ys))]
    # extend_sizes = [extend_size for i in range(len(ys))]
    # modelss = [models for i in range(len(ys))]
    
    # pool = ThreadPool(2)
    # pool.map(append_ecm_to_models, zip(xses_series, ys, modelss, input_sizes, extend_sizes))
    for i in tqdm(range(len(ys))):
        append_ecm_to_models((xses_series[i], ys[i], models, input_size, extend_size))

    i = 0
    for model in models:
        model.display_plot(save_path=plots_dir / f"trained{i}.png")
        i+=1

    xses_series, ys = loadArff.load_cricket_normalized(test_path)

    test_models = []
    # modelss = [test_models for i in range(len(ys))]
    # pool.map(append_ecm_to_models, zip(xses_series, ys, modelss, input_sizes, extend_sizes))

    for i in tqdm(range(len(ys))):
        append_ecm_to_models((xses_series[i], ys[i], test_models, input_size, extend_size))

    i = 0
    for model in models:
        model.display_plot(save_path=plots_dir / f"tested{i}.png")
        i+=1

    mismatches = 0
    for model in test_models:
        class_prediction = nn_weights(models, model).get_class()
        if class_prediction != model.get_class():
            print(f"Error: {class_prediction} should be {model.get_class()}")
            mismatches += 1
    print(f"Accuracy (weights nn): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")

    mismatches = 0
    for model in test_models:
        class_prediction = nn_weights_and_start_values(models, model).get_class()
        if class_prediction != model.get_class():
            print(f"Error: {class_prediction} should be {model.get_class()}")
            mismatches += 1
    print(f"Accuracy (weights and start values nn): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")


    mismatches = 0
    i = 0
    for model in test_models:
        class_prediction = nn_convergence(models, model, xses_series[i][0]).get_class()
        if class_prediction != model.get_class():
            print(f"Error: {class_prediction} should be {model.get_class()}")
            mismatches += 1
        i+=1
    print(f"Accuracy (convergence nn): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")


if __name__ == "__main__":
    test_ecm_nn()
    # what if we compared points of covergence for these trained ecms?