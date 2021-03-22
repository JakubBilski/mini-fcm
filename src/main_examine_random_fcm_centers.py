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


def get_score_of_fcm_with_centers(train_xses_series, train_ys, test_xses_series, test_ys, centers):
    print(f'\nExecuting for centers:')
    print(centers)
    print(f'Transforming to the centers\' space')

    train_xses_series_transformed = cmeans.transform(
        xses_series=train_xses_series,
        centers=centers)

    test_xses_series_transformed = cmeans.transform(
        xses_series=test_xses_series,
        centers=centers)

    train_models = []

    print(f'Learning train')
    for xs, y in tqdm(zip(train_xses_series_transformed, train_ys)):
        model = DECognitiveMap(no_centers)
        model.train(xs)
        model.set_class(y)
        train_models.append(model)

    test_models = []

    for xs, y in tqdm(zip(test_xses_series_transformed, test_ys)):
        model = DECognitiveMap(no_centers)
        model.set_class(y)
        test_models.append(model)

    print("Example model:")
    print(train_models[0].weights)

    print("Share of degenerated weights:")
    indecency = get_models_indecency(train_models)
    print(indecency)

    print(f'classifying with best_prediction')
    accuracy = accuracyComparing.get_accuracy(
        train_models=train_models,
        test_models=test_models,
        test_xs=test_xses_series_transformed,
        input_size=no_centers,
        no_classes=no_classes,
        classification_method="best_prediction")
    print(f'accuracy: {accuracy}')
    return accuracy, indecency


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

    no_centers = 3

    for dataset_name, no_classes in datasets:
        csv_path = f'plots\\picked\\centers_generated_with_hmm\\{dataset_name}_2_hmm_centers.csv'

        no_random_samples = 2

        print(f"{dataset_name}")
        print(f"no_centers: {no_centers}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_all_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)

        random_centerss = generate_random_centers(no_random_samples, no_centers)
        random_centerss_accuracy = []
        random_centerss_indecency = []

        for random_centers in tqdm(random_centerss):
            acc, ind = get_score_of_fcm_with_centers(
                train_xses_series,
                train_ys,
                test_xses_series,
                test_ys,
                random_centers
            )
            random_centerss_accuracy.append(acc)
            random_centerss_indecency.append(ind)

        print("\nGenerating cmeans centers")

        cmeans_centers, _ = cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers,
            m=2.0)
        
        cmeans_centers_accuracy, cmeans_centers_indecency = get_score_of_fcm_with_centers(
                train_xses_series,
                train_ys,
                test_xses_series,
                test_ys,
                random_centers
            )

        better_than_cmeans = [i for i in range(no_random_samples) if random_centerss_accuracy[i] > cmeans_centers_accuracy]
        print(f"{100*len(better_than_cmeans)/no_random_samples}% of centers performed better than cmeans")
        print(f"cmeans accuracy: {cmeans_centers_accuracy}")
        print(f"cmeans indecency: {cmeans_centers_indecency}")
        print(f"mean random accuracy: {np.mean(random_centerss_accuracy)}")
        print(f"mean random indecency: {np.mean(random_centerss_indecency)}")
        print(f"mean indecency when more accurate: {np.mean([random_centerss_indecency[i] for i in better_than_cmeans])}")
        indecency_zero = [i for i in range(no_random_samples) if random_centerss_indecency[i] == 0]
        print(f"mean random accuracy when indecency zero: {np.mean([random_centerss_accuracy[i] for i in indecency_zero])}")

        correlation_acc_ind = np.corrcoef(random_centerss_accuracy, random_centerss_indecency)[0, 1]
        print(f"correlation of accuracy and indecency {correlation_acc_ind}")

