from tqdm import tqdm
from datetime import datetime
import numpy as np
import json
import copy
import os
# from multiprocessing.dummy import Pool as ThreadPool

import pathlib
from examiningConvergence.examineConvergence import examineConvergence

from loadingData import loadArff
from cognitiveMaps import consts
from cognitiveMaps import comparing
from cognitiveMaps.ecmTrainingPath import ECMTrainingPath
from cognitiveMaps.extendedCognitiveMap import ExtendedCognitiveMap


train_path = pathlib.Path('./data/Cricket/Cricket_TRAIN.arff')
test_path = pathlib.Path('./data/Cricket/Cricket_TEST.arff')
# train_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff')
# test_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')
checkpoints_dir = pathlib.Path(f'./checkpoints/Cricket/')


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
        fcm = FuzzyCognitiveMap(input_size)
        fcm.train(xses_series[i])
        class_prediction = comparing.nn_weights(models, fcm).get_class()
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
    fcm.train(xs, steps=100)
    fcm.set_class(y)
    models.append(fcm)


def test_ecm_nn():
    os.mkdir(plots_dir)
    input_size = 3
    extend_size = 3
    models = []
    xses_series, ys = loadArff.load_uwave_normalized(train_path)
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

    xses_series, ys = loadArff.load_uwave_normalized(test_path)

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
        class_prediction = comparing.nn_weights(models, model).get_class()
        if class_prediction != model.get_class():
            print(f"Error: {class_prediction} should be {model.get_class()}")
            mismatches += 1
    print(f"Accuracy (weights nn): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")

    mismatches = 0
    for model in test_models:
        class_prediction = comparing.nn_weights_and_start_values(models, model, input_size, extend_size).get_class()
        if class_prediction != model.get_class():
            print(f"Error: {class_prediction} should be {model.get_class()}")
            mismatches += 1
    print(f"Accuracy (weights and start values nn): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")


def small_steps_ecn_nn():
    print("Start")
    os.mkdir(plots_dir)
    input_size = 6
    extend_size = 3
    models = []
    xses_series, ys = loadArff.load_cricket_normalized(train_path)
    # xses_series = xses_series[0:len(ys):9]
    # ys = ys[0:len(ys):9]
    ys = [int(y-1) for y in ys]
    no_groups = 12
    # input_sizes = [input_size for i in range(len(ys))]
    # extend_sizes = [extend_size for i in range(len(ys))]
    # modelss = [models for i in range(len(ys))]
    
    # pool = ThreadPool(2)
    # pool.map(append_ecm_to_models, zip(xses_series, ys, modelss, input_sizes, extend_sizes))
    for i in range(len(ys)):
        model = ExtendedCognitiveMap(input_size, input_size+extend_size)
        model.set_class(ys[i])
        models.append(model)
    
    good_grouping_factor = 0.3
    
    print("Learning models")
    for i in tqdm(range(len(ys))):
        models[i].train_step(xses_series[i])

    grouping_factor = get_grouping_factor(models, input_size, extend_size, no_groups, ys)
    step = 1

    while step < 100:
        print(f"step {step}, grouping factor {grouping_factor}")
        for i in tqdm(range(len(ys))):
            models[i].train_step(xses_series[i])
        grouping_factor = get_grouping_factor(models, input_size, extend_size, no_groups, ys)
        step += 1
    
    xses_series, ys = loadArff.load_cricket_normalized(test_path)

    mismatches = 0
    print("Identifying test cases")
    for xs, y in tqdm(zip(xses_series, ys)):
        class_prediction = comparing.best_prediction(models, xs).get_class()
        if class_prediction != y:
            print(f"Error: {class_prediction} should be {y}")
            mismatches += 1
    print(f"Accuracy (best prediction): {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")
