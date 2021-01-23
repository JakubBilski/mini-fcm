from tqdm import tqdm
import copy
# from multiprocessing.dummy import Pool as ThreadPool

from examiningConvergence.examineConvergence import examineConvergence

from loadingData import loadArff
from .ecmTrainingPath import ECMTrainingPath
from .extendedCognitiveMap import ExtendedCognitiveMap

def create_checkpoints(input_path, output_path, learning_rate, steps, input_size, extended_size):
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    for i in tqdm(range(0,len(ys))):
        training_path = ECMTrainingPath(learning_rate, ys[i])
        ecm = ExtendedCognitiveMap(input_size, input_size+extended_size)
        ecm.set_class(ys[i])
        training_path.points.append(copy.deepcopy(ecm))
        file = open(output_path / f'training_path{i}.json', 'w+')
        for step in range(steps):
            ecm.train_step(xses_series[i], learning_rate)
            training_path.points.append(copy.deepcopy(ecm))
        file.write(training_path.to_json())
        file.close()

def load_full_checkpoints(checkpoints_dir):
    training_paths = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        training_paths.append(ECMTrainingPath.from_json(file.read()))
    print(training_paths[0].points[0].weights)
    return training_paths


def load_chosen_step_checkpoints(checkpoints_dir, chosen_step = -1):
    models = []
    for file_path in checkpoints_dir.iterdir():
        file = open(file_path, 'r')
        models.append(ECMTrainingPath.from_json_chosen_step(file.read(), -1))
    return models