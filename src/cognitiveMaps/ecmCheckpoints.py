import json
import copy
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from loadingData import loadArff
from .extendedCognitiveMap import ExtendedCognitiveMap

USE_MULTIPROCESSING = False

class ECMTrainingPath:
    def __init__(self, learning_rate, class_name, input_data_index) -> None:
        self.points = []
        self.learning_rate = learning_rate
        self.class_name = class_name
        self.input_data_index = input_data_index
    
    def to_json(self):
        d = {}
        d['weights'] = [p.weights.tolist() for p in self.points]
        d['start_values'] = [p.start_values.tolist() for p in self.points]
        d['learning_rate'] = self.learning_rate
        d['class_name'] = self.class_name
        d['input_data_index'] = self.input_data_index
        d['k'] = self.points[0].k
        d['n'] = self.points[0].n
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        learning_rate = d['learning_rate']
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        k = d['k']
        training_path = ECMTrainingPath(learning_rate, class_name, input_data_index)
        for ws, sv in zip(d['weights'], d['start_values']):
            ecm = ExtendedCognitiveMap(k, n)
            ecm.weights = np.asarray(ws) 
            ecm.start_values = np.asarray(sv)
            ecm.set_class(class_name)
            training_path.points.append(ecm)
        return training_path

    def from_json_chosen_step(source, step):
        d = json.loads(source)
        class_name = d['class_name']
        input_data_index = d['input_data_index']
        n = d['n']
        k = d['k']
        ecm = ExtendedCognitiveMap(k, n)
        ecm.weights = np.asarray(d['weights'][step]) 
        ecm.start_values = np.asarray(d['start_values'][step])
        ecm.set_class(class_name)
        ecm.n = n
        ecm.k = k
        return ecm, input_data_index

def create_checkpoints(input_path, output_path, learning_rate, steps, input_size, extended_size):
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    config = (learning_rate, steps, input_size, extended_size)
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
    learning_rate, steps, input_size, extended_size = config
    training_path = ECMTrainingPath(learning_rate, y, i)
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


def load_checkpoints_chosen_step(checkpoints_dir, chosen_step = -1):
    models = []
    # os.mkdir(plots_dir)
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        ecm, input_data_index = ECMTrainingPath.from_json_chosen_step(file.read(), -1)
        models.append((ecm, input_data_index))
        # ecm.display_plot(plots_dir / f"{file_path.name}.png")
    return models