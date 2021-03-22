# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from cognitiveMaps.hmm import HMM
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadSktime
from examiningData import displaying

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def generate_random_centers(no_samples, no_centers):
    np.random.seed(1)
    centerss = [np.asarray([np.random.rand(2) for _1 in range(no_centers)]) for _ in range(no_samples)]
    return centerss


def use_centers_to_train_models(train_xses_series, train_ys, centers):
    print(f'\nExecuting for centers:')
    print(centers)
    print(f'Transforming to the centers\' space')

    train_xses_series_transformed = cmeans.transform(
        xses_series=train_xses_series,
        centers=centers)

    train_models = []

    print(f'Learning train')
    for xs, y in tqdm(zip(train_xses_series_transformed, train_ys)):
        model = DECognitiveMap(no_centers)
        model.train(xs)
        model.set_class(y)
        train_models.append(model)

    print("Example model:")
    print(train_models[0].weights)

    print("Share of degenerated weights:")
    indecency = get_models_indecency(train_models)
    print(indecency)

    return train_models, indecency

def get_classification_score(
    train_models,
    test_xses_series,
    test_ys,
    centers):

    print(f'Transforming test data to the centers\' space')

    test_xses_series_transformed = cmeans.transform(test_xses_series, centers)
    test_models = []

    for xs, y in zip(test_xses_series_transformed, test_ys):
        model = DECognitiveMap(no_centers)
        model.set_class(y)
        test_models.append(model)

    print(f'classifying with best_prediction')
    accuracy = accuracyComparing.get_accuracy(
        train_models=train_models,
        test_models=test_models,
        test_xs=test_xses_series_transformed,
        input_size=no_centers,
        no_classes=no_classes,
        classification_method="best_prediction")
    print(f'accuracy: {accuracy}')
    return accuracy


def get_models_indecency(models):
    threshold = 0.99
    no_degenerated_weights = 0
    for model in models:
        no_degenerated_weights += np.sum(model.weights >= threshold)
        no_degenerated_weights += np.sum(model.weights <= -threshold)
    return no_degenerated_weights/(models[0].n*models[0].n*len(models))


def load_all_data(
    data_loading_function,
    test_path,
    train_path,
    derivative_order):

    print("Loading data")
    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    print("Normalizing")
    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    return train_xses_series, train_ys, test_xses_series, test_ys


if __name__ == "__main__":

    os.mkdir(plots_dir)

    datasets = [
        ('ACSF1', 10),
        ('Adiac', 37),
        ('ArrowHead', 3),
        ('Beef', 5),
        ('BeetleFly', 2),
        ('BirdChicken', 2),
        ('BME', 3),
        ('Car', 4),
        ('CBF', 3),
        ('Chinatown', 2),
        ('ChlorineConcentration', 3),
        ('CinCECGTorso', 4),
        ('Coffee', 2),
        ('Computers', 2),
        ('CricketX', 12),
        ('CricketY', 12),
        ('CricketZ', 12),
        ('Crop', 24),
        ('DiatomSizeReduction', 4),
        ('DistalPhalanxOutlineAgeGroup', 3),
        ('DistalPhalanxOutlineCorrect', 2),
        ('DistalPhalanxTW', 6),
        ('DodgerLoopDay', 7),
        ('DodgerLoopGame', 2),
        ('DodgerLoopWeekend', 2),
        ('Earthquakes', 2),
        ('ECG200', 2),
        ('ECG5000', 5),
        ('ECGFiveDays', 2),
        ('ElectricDevices', 7),
        ('EOGHorizontalSignal', 12),
        ('EOGVerticalSignal', 12),
        ('EthanolLevel', 4),
        ('FaceAll', 14),
        ('FaceFour', 4),
        ('FacesUCR', 14),
        ('FiftyWords', 50),
        ('Fish', 7),
        ('FordA', 2),
        ('FordB', 2),
        ('FreezerRegularTrain', 2),
        ('FreezerSmallTrain', 2),
        ('Fungi', 18),
        ('GestureMidAirD1', 26),
        ('GestureMidAirD2', 26),
        ('GestureMidAirD3', 26),
        ('GunPoint', 2),
        ('GunPointAgeSpan', 2),
        ('GunPointMaleVersusFemale', 2),
        ('GunPointOldVersusYoung', 2),
        ('Ham', 2),
        ('HandOutlines', 2),
        ('Haptics', 5),
        ('Herring', 2),
        ('HouseTwenty', 2),
        ('InlineSkate', 7),
        ('InsectEPGRegularTrain', 3),
        ('InsectEPGSmallTrain', 3),
        ('ItalyPowerDemand', 2),
        ('LargeKitchenAppliances', 3),
        ('Lightning2', 2),
        ('Lightning7', 7),
        ('Mallat', 8),
        ('Meat', 3),
        ('MedicalImages', 10),
        ('MelbournePedestrian', 10),
        ('MiddlePhalanxOutlineAgeGroup', 3),
        ('MiddlePhalanxOutlineCorrect', 2),
        ('MiddlePhalanxTW', 6),
        ('MixedShapesSmallTrain', 5),
        ('MoteStrain', 2),
        ('NonInvasiveFetalECGThorax1', 42),
        ('NonInvasiveFetalECGThorax2', 42),
        ('OliveOil', 4),
        ('OSULeaf', 6),
        ('PhalangesOutlinesCorrect', 2),
        ('Phoneme', 39),
        ('PigAirwayPressure', 52),
        ('PigArtPressure', 52),
        ('PigCVP', 52),
        ('Plane', 7),
        ('PowerCons', 2),
        ('ProximalPhalanxOutlineAgeGroup', 3),
        ('ProximalPhalanxOutlineCorrect', 2),
        ('ProximalPhalanxTW', 6),
        ('RefrigerationDevices', 3),
        ('Rock', 4),
        ('ScreenType', 3),
        ('SemgHandGenderCh2', 2),
        ('SemgHandMovementCh2', 6),
        ('SemgHandSubjectCh2', 5),
        ('ShapeletSim', 2),
        ('ShapesAll', 60),
        ('SmallKitchenAppliances', 3),
        ('SmoothSubspace', 3),
        ('SonyAIBORobotSurface1', 2),
        ('SonyAIBORobotSurface2', 2),
        ('StarLightCurves', 3),
        ('Strawberry', 2),
        ('SwedishLeaf', 15),
        ('Symbols', 6),
        ('SyntheticControl', 6),
        ('ToeSegmentation1', 2),
        ('ToeSegmentation2', 2),
        ('Trace', 4),
        ('TwoLeadECG', 2),
        ('TwoPatterns', 4),
        ('UMD', 3),
        ('UWaveGestureLibraryAll', 8),
        ('UWaveGestureLibraryX', 8),
        ('UWaveGestureLibraryY', 8),
        ('UWaveGestureLibraryZ', 8),
        ('Wafer', 2),
        ('Wine', 2),
        ('WordSynonyms', 25),
        ('Worms', 5),
        ('WormsTwoClass', 2),
        ('Yoga', 2),
    ]

    datasets = [datasets[0]]

    range_centers = [2,3]
    no_random_samples = 1

    for dataset_name, no_classes in datasets:
        csv_path = plots_dir / f'{dataset_name}_results.csv'
        csv_results_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_results_file)
        csv_writer.writerow(['dataset', 'no_classes', 'method', 'no_random_samples'])
        csv_writer.writerow([dataset_name, no_classes, 'random center with lowest indecency', no_random_samples])
        csv_writer.writerow(['no_centers', 'accuracy'])
        print(f"{dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_all_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)

        for no_centers in range_centers:
            print(f"\nno_centers: {no_centers}")
            random_centerss = generate_random_centers(no_random_samples, no_centers)
            chosen_models = None
            chosen_centers = None
            chosen_models_indecency = 1.01

            for random_centers in random_centerss:
                train_models, ind = use_centers_to_train_models(
                    train_xses_series,
                    train_ys,
                    random_centers
                )
                if ind < chosen_models_indecency:
                    chosen_models = train_models
                    chosen_centers = random_centers
                    chosen_models_indecency = ind
                    if chosen_models_indecency == 0:
                        print("Found solution with indecency 0.0")
                        break
            
            print("\nGenerating cmeans centers")

            accuracy = get_classification_score(
                    chosen_models,
                    test_xses_series,
                    test_ys,
                    chosen_centers
                )
            
            csv_writer.writerow([no_centers, accuracy])
        csv_results_file.close()
