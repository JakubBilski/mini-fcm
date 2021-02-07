import json
import copy
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from loadingData import loadArff
from .extendedCognitiveMap import ExtendedCognitiveMap

USE_MULTIPROCESSING = False


class ECMTrainingPath:
    def __init__(self, learning_rate, class_name,
                 input_data_index, cmeans_centers=None) -> None:
        self.points = []
        self.learning_rate = learning_rate
        self.class_name = class_name
        self.input_data_index = input_data_index
        self.cmeans_centers = cmeans_centers

    def to_json(self):
        d = {}
        d['weights'] = [p.weights.tolist() for p in self.points]
        d['start_values'] = [p.start_values.tolist() for p in self.points]
        d['learning_rate'] = self.learning_rate
        d['class_name'] = self.class_name
        d['input_data_index'] = self.input_data_index
        d['k'] = self.points[0].k
        d['n'] = self.points[0].n
        if self.cmeans_centers is not None:
            d['cmeans_centers'] = self.cmeans_centers
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        learning_rate = d['learning_rate']
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        k = d['k']
        if 'cmeans_centers' in d.keys():
            cmeans_centers = d['cmeans_centers']
        else:
            cmeans_centers = None
        training_path = ECMTrainingPath(
            learning_rate,
            class_name,
            input_data_index,
            cmeans_centers)
        for ws, sv in zip(d['weights'], d['start_values']):
            ecm = ExtendedCognitiveMap(k, n)
            ecm.weights = np.asarray(ws)
            ecm.start_values = np.asarray(sv)
            ecm.set_class(class_name)
            training_path.points.append(ecm)
        return training_path


def create_checkpoints(xses_series, ys, output_path, learning_rate, steps,
                       input_size, extended_size, save_step=1, cmeans_centers=None):
    config = (learning_rate, steps, input_size, extended_size, cmeans_centers, save_step)
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
    learning_rate, steps, input_size, extended_size, cmeans_centers, save_step = config
    training_path = ECMTrainingPath(learning_rate, y, i, cmeans_centers)
    ecm = ExtendedCognitiveMap(input_size, input_size+extended_size)
    ecm.set_class(y)
    training_path.points.append(copy.deepcopy(ecm))
    for step in range(steps):
        ecm.train_step(xs, learning_rate)
        if step % save_step == 0 or step == steps-1:
            training_path.points.append(copy.deepcopy(ecm))
    return training_path


def load_checkpoints(checkpoints_dir):
    training_paths = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        training_paths.append(ECMTrainingPath.from_json(file.read()))
    return training_paths
