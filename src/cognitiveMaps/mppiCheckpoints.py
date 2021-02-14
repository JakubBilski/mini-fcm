import json
import copy
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from .mppiCognitiveMap import MppiCognitiveMap

USE_MULTIPROCESSING = False


class MPPICheckpoint:
    def __init__(self, mppi, input_data_index, cmeans_centers=None) -> None:
        self.mppi = mppi
        self.weights = mppi.weights
        self.class_name = mppi.get_class()
        self.input_data_index = input_data_index
        self.n = mppi.n
        self.cmeans_centers = cmeans_centers

    def to_json(self):
        d = {}
        d['weights'] = self.weights.tolist()
        d['class_name'] = self.class_name
        d['input_data_index'] = self.input_data_index
        d['n'] = self.n
        if self.cmeans_centers is not None:
            d['cmeans_centers'] = self.cmeans_centers
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        class_name = d['class_name']
        weights = d['weights']
        input_data_index = d['input_data_index']
        n = d['n']
        if 'cmeans_centers' in d.keys():
            cmeans_centers = d['cmeans_centers']
        else:
            cmeans_centers = None
        mppi = MppiCognitiveMap(n)
        mppi.weights = np.asarray(weights)
        mppi.set_class(class_name)

        checkpoint = MPPICheckpoint(
            mppi,
            input_data_index,
            cmeans_centers)

        return checkpoint

def create_checkpoints(xses_series, ys, output_path, input_size, cmeans_centers=None):
    config = (input_size, cmeans_centers)
    configs = [(config, i) for i in range(len(ys))]

    if USE_MULTIPROCESSING:
        pool = ThreadPool()
        unique_file_id = 0

        for training_path in tqdm(pool.imap_unordered(
                _create_checkpoint, zip(xses_series, ys, configs))):
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
            training_path = _create_checkpoint(traning_path_input)
            file = open(
                output_path / f'training_path{unique_file_id}.json',
                'w+')
            unique_file_id += 1
            file.write(training_path.to_json())
            file.close()

def _create_checkpoint(args):
    xs, y, config = args
    config, i = config
    input_size, cmeans_centers = config
    mppi = MppiCognitiveMap(input_size)
    mppi.train(xs)
    mppi.set_class(y)

    checkpoint = MPPICheckpoint(
        mppi,
        i,
        cmeans_centers)

    return checkpoint


def load_checkpoints(checkpoints_dir):
    checkpoints = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        checkpoints.append(MPPICheckpoint.from_json(file.read()))
    return checkpoints
