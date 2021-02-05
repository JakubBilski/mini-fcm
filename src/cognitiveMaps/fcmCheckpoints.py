import json
import copy
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from loadingData import loadArff
from .fuzzyCognitiveMap import FuzzyCognitiveMap

USE_MULTIPROCESSING = False

class FCMTrainingPath:
    def __init__(self, learning_rate, class_name, input_data_index, input_size) -> None:
        self.points = []
        self.learning_rate = learning_rate
        self.class_name = class_name
        self.input_data_index = input_data_index
        self.n = input_size

    def to_json(self):
        d = {}
        d['weights'] = [p.weights.tolist() for p in self.points]
        d['learning_rate'] = self.learning_rate
        d['class_name'] = self.class_name
        d['input_data_index'] = self.input_data_index
        d['n'] = self.n
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        learning_rate = d['learning_rate']
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        training_path = FCMTrainingPath(learning_rate, class_name, input_data_index, n)
        for ws in d['weights']:
            fcm = FuzzyCognitiveMap(n)
            fcm.weights = np.asarray(ws)
            fcm.set_class(class_name)
            training_path.points.append(fcm)
        return training_path

    def from_json_chosen_step(source, step):
        d = json.loads(source)
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        fcm = FuzzyCognitiveMap(n)
        fcm.weights = np.asarray(d['weights'][step])
        fcm.set_class(class_name)
        return fcm, input_data_index

def create_checkpoints(input_path, output_path, learning_rate, steps, input_size):
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    config = (learning_rate, steps, input_size)
    configs = [(config, i) for i in range(len(ys))]
    
    if USE_MULTIPROCESSING:
        pool = ThreadPool()
        unique_file_id = 0

        for training_path in tqdm(pool.imap_unordered(_create_training_path, zip(xses_series, ys, configs))):
            file = open(output_path / f'training_path{unique_file_id}.json', 'w+')
            unique_file_id += 1
            file.write(training_path.to_json())
            file.close()
        pool.close()
    else:
        unique_file_id = 0
        for traning_path_input in tqdm(zip(xses_series, ys, configs)):
            training_path = _create_training_path(traning_path_input)
            file = open(output_path / f'training_path{unique_file_id}.json', 'w+')
            unique_file_id += 1
            file.write(training_path.to_json())
            file.close()


def _create_training_path(args):
    xs, y, config = args
    config, i = config
    learning_rate, steps, input_size = config
    training_path = FCMTrainingPath(learning_rate, y, i, input_size)
    fcm = FuzzyCognitiveMap(input_size)
    fcm.set_class(y)
    training_path.points.append(copy.deepcopy(fcm))
    for step in range(steps):
        fcm.train_step(xs, learning_rate)
        training_path.points.append(copy.deepcopy(fcm))
    return training_path


def load_checkpoints(checkpoints_dir):
    training_paths = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        training_paths.append(FCMTrainingPath.from_json(file.read()))
    return training_paths


def load_checkpoints_chosen_step(checkpoints_dir, chosen_step = -1):
    models = []
    # os.mkdir(plots_dir)
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        ecm, input_data_index = FCMTrainingPath.from_json_chosen_step(file.read(), -1)
        models.append((ecm, input_data_index))
        # ecm.display_plot(plots_dir / f"{file_path.name}.png")
    return models