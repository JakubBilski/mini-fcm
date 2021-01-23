from sklearn.cluster import KMeans
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


def get_grouping_factor(models, input_size, extend_size, no_clusters, ys):
    vects = [model.weights.flatten().tolist() for model in models]
    for i in range(len(models)):
        vects[i] = np.append(vects[i],
            consts.E(input_size+extend_size, input_size).dot(models[i].start_values).flatten().tolist())

    centers = np.zeros(shape=(no_clusters, len(vects[0])))
    no_center_members = np.zeros(shape=(no_clusters))
    for i in range(len(vects)):
        cluster_class = int(models[i].get_class())
        centers[cluster_class] += vects[i]
        no_center_members[cluster_class] += 1
    for c in range(centers.shape[0]):
        centers[c] /= no_center_members[c]


    kmeans = KMeans(n_clusters=no_clusters, init=centers).fit(vects)
        
    print(kmeans.labels_)
    print(ys)

    return sum(kmeans.labels_ == ys)/len(models)


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

def create_checkpoints():
    learning_rate = 0.002
    steps = 1
    input_size = 6
    extended_size = 3

    if not checkpoints_dir.is_dir():
        os.mkdir(checkpoints_dir)
    checkpoints_path = checkpoints_dir / 'train/'
    os.mkdir(checkpoints_path)

    training_paths = []
    xses_series, ys = loadArff.load_cricket_normalized(train_path)
    for i in tqdm(range(0,len(ys))):
        training_path = ECMTrainingPath(learning_rate, ys[i])
        ecm = ExtendedCognitiveMap(input_size, input_size+extended_size)
        ecm.set_class(ys[i])
        training_path.points.append(copy.deepcopy(ecm))
        for step in range(steps):
            ecm.train_step(xses_series[i], learning_rate)
            training_path.points.append(copy.deepcopy(ecm))
        training_paths.append(training_path)

    unique_file_suffix = 0
    for tp in training_paths:
        file = open(checkpoints_path / f'training_path{unique_file_suffix}.json', 'w')
        file.write(tp.to_json())
        file.close()
        unique_file_suffix += 1

def load_checkpoints():
    checkpoints_path_train = checkpoints_dir / 'train/'
    training_paths = []
    for file_path in checkpoints_path_train.iterdir():
        file = open(file_path, 'r')
        training_paths.append(ECMTrainingPath.from_json(file.read()))
    print(training_paths[0].points[0].weights)

if __name__ == "__main__":
    # test_ecm_nn()
    create_checkpoints()
    load_checkpoints()