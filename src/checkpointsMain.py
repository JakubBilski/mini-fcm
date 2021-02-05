from tqdm import tqdm
import numpy as np
import pathlib
import os

from cognitiveMaps import ecmCheckpoints
from cognitiveMaps import fcmCheckpoints
from cognitiveMaps import displaying
from cognitiveMaps import comparing
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from loadingData import loadArff


def compare_solutions(train_models, test_models, test_xs, test_ys, input_size, extend_size, no_classes):
    mistakes = 0
    for test_model, input_data_index in test_models:
        train_models_without_same = [m for m, _ in train_models if (m.weights != test_model.weights).any()]
        best_fit, best_cost = comparing.nn_weights(train_models_without_same, test_model, input_size+extend_size, input_size)
        if best_fit.get_class() != test_model.get_class():
            good_predictions = [m for m in train_models_without_same if m.get_class()==test_model.get_class()]
            best_correct_fit, best_correct_cost = comparing.nn_weights(good_predictions, test_model, input_size+extend_size, input_size)
            # displaying.draw_cognitive_maps(
            #     [
            #         best_fit.weights,
            #         test_model.weights,
            #         best_correct_fit.weights
            #     ],
            #     [
            #         f"best fit (class {best_fit.get_class()}) (cost {best_cost})",
            #         f"test model (class {test_model.get_class()})",
            #         f"best correct fit (cost {best_correct_cost})"
            #     ]
            # )
            # print(f"{best_fit.get_class()} should be {test_model.get_class()}")
            # print(f"{best_fit} won")
            mistakes += 1
    print(f"nn_weights accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    if extend_size > 0:
        mistakes = 0
        for test_model, input_data_index in test_models:
            train_models_without_same = [m for m, _ in train_models if (m.weights != test_model.weights).any()]
            best_fit = comparing.nn_weights_and_start_values(train_models_without_same, test_model, input_size, extend_size)
            if best_fit.get_class() != test_model.get_class():
                mistakes += 1
        print(f"nn_weights_and_start_values accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    # mistakes = 0
    # for test_model, xs in tqdm(zip(test_models, test_xs)):
    #     best_fit = comparing.best_prediction(train_models, xs)
    #     if best_fit.get_class() != test_model.get_class():
    #         mistakes += 1
    # print(f"best_prediction accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    mistakes = 0
    for test_model, input_data_index in test_models:
        train_models_without_same = [m for m, _ in train_models if (m.weights != test_model.weights).any()]
        best_fit = comparing.nn_convergence(train_models_without_same, test_model, test_xs[input_data_index][0])
        if best_fit.get_class() != test_model.get_class():
            mistakes += 1
    print(f"nn_convergence accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")



    mistakes = 0
    for test_model, input_data_index in test_models:
        train_models_without_same = [m for m, _ in train_models if (m.weights != test_model.weights).any()]
        fit_class, _ = comparing.best_mse_sum(train_models_without_same, test_model, no_classes)
        if fit_class != test_model.get_class():
            mistakes += 1
    print(f"best_mse_sum accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")


def generate_ecm_checkpoints():
    extended_size = 1
    learning_rate = 0.002
    steps = 2
    input_size = 6
    # os.mkdir('./checkpoints/ecm/BasicMotions/')
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/ecm/Cricket/{extended_size}_{learning_rate}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    ecmCheckpoints.create_checkpoints(
        input_path,
        output_path,
        learning_rate,
        steps,
        input_size,
        extended_size
    )


def generate_fcm_checkpoints():
    learning_rate = 0.002
    steps = 50
    input_size = 6
    # os.mkdir('./checkpoints/ecm/BasicMotions/')
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/fcm/Cricket/{learning_rate}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    fcmCheckpoints.create_checkpoints(
        input_path,
        output_path,
        learning_rate,
        steps,
        input_size
    )

if __name__ == "__main__":

    # generate_fcm_checkpoints()

    # # solution comparison for ecms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/test')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_training_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_training_paths = ecmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_training_paths[0].points)):
    #     print(f"Step {step}")
    #     train_models = [tp.points[step] for tp in train_training_paths]
    #     test_models = [tp.points[step] for tp in test_training_paths]
    #     compare_solutions(test_models, test_models, xses_series, None, 6, 3, 12)

    # # examine grouping factor based on k-means
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/test')
    # train_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_paths = ecmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_paths[0].points)):
    #     train_models = [train_path.points[step] for train_path in train_paths]
    #     test_models = [test_path.points[step] for test_path in test_paths]
    #     print(f"step {step}")
    #     print(f"train grouping factor: {comparing.get_grouping_factor(train_models, 6, 3, 12)}")
    #     print(f"test grouping factor: {comparing.get_grouping_factor(test_models, 6, 3, 12)}")
    #     print(f"all grouping factor: {comparing.get_grouping_factor(test_models+train_models, 6, 3, 12)}")

    # # examine error of ecms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # for train_path in train_paths:
    #     print(f"New train path (class {train_path.class_name})")
    #     if train_path.class_name != ys[train_path.input_data_index]:
    #         print("Mismatch!")
    #     print([train_path.points[step].get_error(xses_series[train_path.input_data_index])
    #         for step in range(len(train_path.points))])

    
    # # examine error of fcms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/fcm/Cricket/0.002/train')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_paths = fcmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # for train_path in train_paths:
    #     print(f"New train path (class {train_path.class_name})")
    #     if train_path.class_name != ys[train_path.input_data_index]:
    #         print("Mismatch!")
    #     print([train_path.points[step].get_error(xses_series[train_path.input_data_index])
    #         for step in range(len(train_path.points))])

    # # solution comparison for fcms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/fcm/Cricket/0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/fcm/Cricket/0.002/test')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_training_paths[0].points)):
    #     print(f"Step {step}")
    #     train_models = [(tp.points[step], tp.input_data_index) for tp in train_training_paths]
    #     test_models = [(tp.points[step], tp.input_data_index) for tp in test_training_paths]
    #     compare_solutions(test_models, test_models, xses_series, None, 6, 0, 12)


    # # solution comparison for mppi
    # train_input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # test_input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # train_xses_series, train_ys = loadArff.load_cricket_normalized(train_input_file)
    # test_xses_series, test_ys = loadArff.load_cricket_normalized(test_input_file)
    # train_models = []
    # input_data_index = 0
    # for xs, y in zip(train_xses_series, train_ys):
    #     mppicm = MppiCognitiveMap(6)
    #     mppicm.train(xs)
    #     mppicm.set_class(y)
    #     train_models.append((mppicm, input_data_index))
    #     input_data_index+=1
    # test_models = []
    # input_data_index = 0
    # for xs, y in zip(test_xses_series, test_ys):
    #     mppicm = MppiCognitiveMap(6)
    #     mppicm.train(xs)
    #     mppicm.set_class(y)
    #     test_models.append((mppicm, input_data_index))
    #     input_data_index+=1
    # compare_solutions(test_models, test_models, test_xses_series, test_ys, 6, 0, 12)
