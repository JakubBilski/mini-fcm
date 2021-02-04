from tqdm import tqdm
import pathlib
from datetime import datetime
import copy
import os
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp

from examiningConvergence.examineConvergence import examineConvergence

from loadingData import loadArff
from .ecmTrainingPath import ECMTrainingPath
from .extendedCognitiveMap import ExtendedCognitiveMap

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def create_checkpoints(input_path, output_path, learning_rate, steps, input_size, extended_size):
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    config = (learning_rate, steps, input_size, extended_size)
    configs = [(config, i) for i in range(len(ys))]
    
    # os.mkdir(plots_dir)
    pool = ThreadPool()
    unique_file_id = 0

    for training_path in tqdm(pool.imap_unordered(create_training_path, zip(xses_series, ys, configs))):
        file = open(output_path / f'training_path{unique_file_id}.json', 'w+')
        unique_file_id += 1
        file.write(training_path.to_json())
        file.close()
    pool.close()

def create_training_path(args):
    xs, y, config = args
    config, i = config
    learning_rate, steps, input_size, extended_size = config
    training_path = ECMTrainingPath(learning_rate, y)
    ecm = ExtendedCognitiveMap(input_size, input_size+extended_size)
    ecm.set_class(y)
    training_path.points.append(copy.deepcopy(ecm))
    for step in range(steps):
        ecm.train_step(xs, learning_rate)
        training_path.points.append(copy.deepcopy(ecm))
    return training_path



def load_checkpoints(checkpoints_dir):
    training_paths = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        training_paths.append(ECMTrainingPath.from_json(file.read()))
    print(training_paths[0].points[0].weights)
    return training_paths


def load_chosen_step_checkpoints(checkpoints_dir, chosen_step = -1):
    models = []
    # os.mkdir(plots_dir)
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        ecm = ECMTrainingPath.from_json_chosen_step(file.read(), -1)
        models.append(ecm)
        # ecm.display_plot(plots_dir / f"{file_path.name}.png")
    return models