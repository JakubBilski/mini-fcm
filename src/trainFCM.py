from tqdm import tqdm
from datetime import datetime
import os

import pathlib
from examiningConvergence.examineConvergence import examineConvergence


from loadingData import loadArff
from cognitiveMaps import cognitiveMap


train_path = pathlib.Path('./data/Cricket/Cricket_TRAIN.arff')
test_path = pathlib.Path('./data/Cricket/Cricket_TEST.arff')
# train_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff')
# test_path = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def weights_distance(weights_a, weights_b):
    result = 0
    for i in range(len(weights_a)):
        for j in range(len(weights_a[0])):
            result += abs(weights_a[i][j]-weights_b[i][j])
    return result

def find_matching_model(models, m):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model

if __name__ == "__main__":
    os.mkdir(plots_dir)
    input_size = 6
    models = []
    xses_series, ys = loadArff.load_cricket_normalized(train_path)
    for i in tqdm(range(0,len(ys))):
        fcm = cognitiveMap.FuzzyCognitiveMap(input_size)
        fcm.train(xses_series[i])
        fcm.set_class(ys[i])
        models.append(fcm)
        # fcm.display_plot(plots_dir / f"trained{i}.png")
        # fcm.display_plot()
        # examineConvergence(fcm)
    
    xses_series, ys = loadArff.load_cricket_normalized(test_path)
    mismatches = 0
    for i in tqdm(range(len(ys))):
        fcm = cognitiveMap.FuzzyCognitiveMap(input_size)
        fcm.train(xses_series[i])
        class_prediction = find_matching_model(models, fcm).get_class()
        if class_prediction != ys[i]:
            mismatches += 1
            # fcm.display_plot(plots_dir / f"predicted{i}_mistake{class_prediction}_{ys[i]}.png")
        else:
            pass
            # fcm.display_plot(plots_dir / f"predicted{i}.png")
    
    print(f"Accuracy: {len(ys)-mismatches}/{len(ys)} ({100*(len(ys)-mismatches)/len(ys)}%)")
    # 86% Cricket
    # 31% UWave
