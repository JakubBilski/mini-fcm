import json
import copy
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from .fuzzyCognitiveMap import FuzzyCognitiveMap

USE_MULTIPROCESSING = False


class FCMTrainingPath:
    def __init__(self, learning_rate, class_name,
                 input_data_index, input_size, cmeans_centers=None) -> None:
        self.points = []
        self.learning_rate = learning_rate
        self.class_name = class_name
        self.input_data_index = input_data_index
        self.n = input_size
        self.cmeans_centers = cmeans_centers

    def to_json(self):
        d = {}
        d['weights'] = [p.weights.tolist() for p in self.points]
        d['learning_rate'] = self.learning_rate
        d['class_name'] = self.class_name
        d['input_data_index'] = self.input_data_index
        d['n'] = self.n
        if self.cmeans_centers is not None:
            d['cmeans_centers'] = self.cmeans_centers
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        learning_rate = d['learning_rate']
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        if 'cmeans_centers' in d.keys():
            cmeans_centers = d['cmeans_centers']
        else:
            cmeans_centers = None
        training_path = FCMTrainingPath(
            learning_rate,
            class_name,
            input_data_index,
            n,
            cmeans_centers)
        for ws in d['weights']:
            fcm = FuzzyCognitiveMap(n)
            fcm.weights = np.asarray(ws)
            fcm.set_class(class_name)
            training_path.points.append(fcm)
        return training_path


def create_checkpoints(xses_series, ys, output_path, learning_rate,
                       steps, input_size, cmeans_centers=None):
    config = (learning_rate, steps, input_size, cmeans_centers)
    configs = [(config, i) for i in range(len(ys))]

    if USE_MULTIPROCESSING:
        pool = ThreadPool()
        unique_file_id = 0

        for training_path in tqdm(pool.imap_unordered(
                _create_training_path, zip(xses_series, ys, configs))):
            file = open(
                output_path / f'training_path{unique_file_id}.json',
                'w+')
            unique_file_id += 1
            file.write(training_path.to_json())
            file.close()
        pool.close()
    else:
        unique_file_id = 0
        for traning_path_input in tqdm(zip(xses_series, ys, configs)):
            training_path = _create_training_path(traning_path_input)
            file = open(
                output_path / f'training_path{unique_file_id}.json',
                'w+')
            unique_file_id += 1
            file.write(training_path.to_json())
            file.close()


def _create_training_path(args):
    xs, y, config = args
    config, i = config
    learning_rate, steps, input_size, cmeans_centers = config
    training_path = FCMTrainingPath(
        learning_rate,
        y,
        i,
        input_size,
        cmeans_centers)
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
