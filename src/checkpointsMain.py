from tqdm import tqdm
import pathlib
import os

from loadingData import loadArff
from cognitiveMaps import comparing
from cognitiveMaps import checkpoints


def compare_solutions(train_models, test_models, test_xs, test_ys, input_size, extend_size):
    mistakes = 0
    for test_model in tqdm(test_models):
        best_fit = comparing.nn_weights(train_models, test_model, input_size+extend_size, input_size)
        if best_fit.get_class() != test_model.get_class():
            mistakes += 1
    print(f"nn_weights accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    mistakes = 0
    for test_model in tqdm(test_models):
        best_fit = comparing.nn_weights_and_start_values(train_models, test_model, input_size, extend_size)
        if best_fit.get_class() != test_model.get_class():
            mistakes += 1
    print(f"nn_weights_and_start_values accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    mistakes = 0
    for test_model, xs in tqdm(zip(test_models, test_xs)):
        best_fit = comparing.best_prediction(train_models, xs)
        if best_fit.get_class() != test_model.get_class():
            mistakes += 1
    print(f"best_prediction accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

def generate_checkpoints():
    extended_size = 3
    learning_rate = 0.002
    steps = 1
    input_size = 6
    
    # os.mkdir('./checkpoints/ecm/BasicMotions/')
    output_path = pathlib.Path(f'./checkpoints/ecm/Cricket/{extended_size}_{learning_rate}')
    # os.mkdir(output_path)
    os.mkdir(output_path / 'train/')
    checkpoints.create_checkpoints(
        './data/Cricket/CRICKET_TRAIN.arff',
        output_path / 'train/',
        learning_rate,
        steps,
        input_size,
        extended_size
    )

if __name__ == "__main__":
    # generate_checkpoints()


    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/BasicMotions/3_0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/BasicMotions/3_0.002/test')
    # input_file = pathlib.Path('./data/BasicMotions/BasicMotions_TEST.arff')
    # train_models = checkpoints.load_chosen_step_checkpoints(checkpoints_train_dir)
    # test_models = checkpoints.load_chosen_step_checkpoints(checkpoints_test_dir)
    # xses_series, ys = loadArff.load_basic_motions(input_file)
    # compare_solutions(train_models, test_models, xses_series, ys, 6, 3)

    checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/test')
    input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    train_models = checkpoints.load_chosen_step_checkpoints(checkpoints_train_dir)
    test_models = checkpoints.load_chosen_step_checkpoints(checkpoints_test_dir)
    xses_series, ys = loadArff.load_cricket_normalized(input_file)
    compare_solutions(train_models, test_models, xses_series, ys, 6, 3)