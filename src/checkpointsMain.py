from tqdm import tqdm
import numpy as np
import pathlib
import os

from loadingData import loadArff
from cognitiveMaps import comparing
from cognitiveMaps import checkpoints
from cognitiveMaps import displaying


def compare_solutions(train_models, test_models, test_xs, test_ys, input_size, extend_size, no_classes):
    mistakes = 0
    for test_model in test_models:
        train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
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

    mistakes = 0
    for test_model in test_models:
        train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
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
    for test_model, xs in zip(test_models, test_xs):
        train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
        best_fit = comparing.nn_convergence(train_models_without_same, test_model, xs[0])
        if best_fit.get_class() != test_model.get_class():
            mistakes += 1
    print(f"nn_convergence accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")



    mistakes = 0
    for test_model in test_models:
        train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
        fit_class, _ = comparing.best_mse_sum(train_models_without_same, test_model, no_classes)
        if fit_class != test_model.get_class():
            mistakes += 1
    print(f"best_mse_sum accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")



def generate_checkpoints():
    extended_size = 3
    learning_rate = 0.002
    steps = 10
    input_size = 6
    
    # os.mkdir('./checkpoints/ecm/BasicMotions/')
    output_path = pathlib.Path(f'./checkpoints/ecm/Cricket/{extended_size}_{learning_rate}')
    # os.mkdir(output_path)
    os.mkdir(output_path / 'test/')
    checkpoints.create_checkpoints(
        './data/Cricket/CRICKET_TEST.arff',
        output_path / 'test/',
        learning_rate,
        steps,
        input_size,
        extended_size
    )

if __name__ == "__main__":

    generate_checkpoints()

    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/BasicMotions/3_0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/BasicMotions/3_0.002/test')
    # input_file = pathlib.Path('./data/BasicMotions/BasicMotions_TEST.arff')
    # train_models = checkpoints.load_chosen_step_checkpoints(checkpoints_train_dir)
    # test_models = checkpoints.load_chosen_step_checkpoints(checkpoints_test_dir)
    # xses_series, ys = loadArff.load_basic_motions(input_file)
    # compare_solutions(train_models, test_models, xses_series, ys, 6, 3)

    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/test')
    # # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # train_models = checkpoints.load_chosen_step_checkpoints(checkpoints_train_dir)
    # test_models = checkpoints.load_chosen_step_checkpoints(checkpoints_test_dir)
    # # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # compare_solutions(test_models, train_models, None, None, 6, 3, 12)

    # # solution comparison step by step
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/test')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_training_paths = checkpoints.load_checkpoints(checkpoints_train_dir)
    # test_training_paths = checkpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_training_paths[0].points)):
    #     print(f"Step {step}")
    #     train_models = [tp.points[step] for tp in train_training_paths]
    #     test_models = [tp.points[step] for tp in test_training_paths]
    #     compare_solutions(test_models, test_models, xses_series, None, 6, 3, 12)

    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/test')
    # train_paths = checkpoints.load_checkpoints(checkpoints_train_dir)
    # test_paths = checkpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_paths[0].points)):
    #     train_models = [train_path.points[step] for train_path in train_paths]
    #     test_models = [test_path.points[step] for test_path in test_paths]
    #     print(f"step {step}")
    #     print(f"train grouping factor: {comparing.get_grouping_factor(train_models, 6, 3, 12)}")
    #     print(f"test grouping factor: {comparing.get_grouping_factor(test_models, 6, 3, 12)}")
    #     print(f"all grouping factor: {comparing.get_grouping_factor(test_models+train_models, 6, 3, 12)}")


    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_paths = checkpoints.load_checkpoints(checkpoints_train_dir)
    # for train_path, xs in zip(train_paths, xses_series):
    #     print(f"New train path (class {train_path.class_name})")
    #     print([train_path.points[step].get_error(xs) for step in range(len(train_path.points))])
