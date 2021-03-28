# flake8: noqa
from tqdm import tqdm
import pathlib
import os
import csv
from datetime import datetime
import argparse

from cognitiveMaps.deCognitiveMap import DECognitiveMap
from cognitiveMaps.deShrinkedCognitiveMap import DEShrinkedCognitiveMap
from cognitiveMaps.deVeryShrinkedCognitiveMap import DEVeryShrinkedCognitiveMap
from cognitiveMaps.hmm import HMM
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from loadingData import loadSktime
from examiningData import displaying
from examiningData import basicExamining

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def test_solution(
    solution_name,
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path):
    
    if solution_name in ['hmm nn', 'hmm 1c']:
        model_class = HMM
    elif solution_name in ['fcm nn', 'fcm 1c']:
        model_class = DECognitiveMap
    elif solution_name == 'sfcm nn':
        model_class = DEShrinkedCognitiveMap
    elif solution_name == 'vsfcm nn':
        model_class = DEVeryShrinkedCognitiveMap
    else:
        raise Exception(f"Solution name {solution_name} not recognized")

    print(f"Starting: {dataset_name} with {solution_name}")

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(['dataset', 'no_classes', 'method'])
    csv_writer.writerow([dataset_name, no_classes, solution_name])
    csv_writer.writerow(['no_states', 'accuracy', 'degenerated_share'])

    for no_states in range(2, 11):
        print(f"no states {no_states}")

        if solution_name in ['fcm nn', 'fcm 1c', 'sfcm nn', 'vsfcm nn']:
            print(f'transforming with cmeans')
            centers, transformed_train_xses_series = cmeans.find_centers_and_transform(
                xses_series=train_xses_series,
                c=no_states)
            transformed_test_xses_series = cmeans.transform(
                xses_series=test_xses_series,
                centers=centers)
        else:
            transformed_train_xses_series = train_xses_series
            transformed_test_xses_series = test_xses_series

        if solution_name in ['hmm 1c', 'fcm 1c']:
            learning_input = [([], i) for i in range(no_classes)]
            for xs, y in zip(transformed_train_xses_series, train_ys):
                learning_input[y][0].append(xs)
        else:
            learning_input = [([xs], y) for xs, y in zip(transformed_train_xses_series, train_ys)]

        print(f'learning train models')
        error_occured = False
        train_models = []
        for i in tqdm(range(len(learning_input))):
            model = model_class(no_states)
            try:
                model.train(learning_input[i][0])
            except:
                error_occured = True
                break
            model.set_class(learning_input[i][1])
            train_models.append(model)
        
        degenerated_share = basicExamining.get_share_of_degenerated_weights(train_models, 0.99)

        if not error_occured:
            try:
                print(f'classifying with best_prediction')
                if solution_name in ['hmm nn', 'hmm 1c']:
                    accuracy = accuracyComparing.get_accuracy_hmm_best_prediction(
                        train_models=train_models,
                        test_xs=transformed_test_xses_series,
                        test_classes=test_ys
                    )
                else:
                    accuracy = accuracyComparing.get_accuracy_fcm_best_prediction(
                        train_models=train_models,
                        test_xs=transformed_test_xses_series,
                        test_classes=test_ys
                    )
            except:
                error_occured = True

        if error_occured:
            print(f"error occured for no_states {no_states}")
            break
        else:
            csv_writer.writerow([no_states, accuracy, degenerated_share])
            print(f'accuracy: {accuracy}')
            print(f'share of degenerated weights: {degenerated_share}')
    
    csv_results_file.close()


def load_preprocessed_data(
    data_loading_function,
    test_path,
    train_path,
    derivative_order,
    dataset_name):

    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    displaying.display_series(
        train_xses_series,
        plots_dir / f'{dataset_name} visualization.png',
        f'{dataset_name} visualization')
    
    return train_xses_series, train_ys, test_xses_series, test_ys



def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification accuracy on TimeSeriesClassification datasets')
    parser.add_argument('--solution',
                        '-s',
                        choices=['sfcm nn', 'hmm nn', 'fcm 1c', 'hmm 1c', 'fcm nn', 'vsfcm nn'],
                        default='sfcm nn',
                        help='How models used during classification will be trained',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    solution_name = args.solution

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

    for dataset_name, no_classes in datasets:
        print(f"Preprocessing {dataset_name}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_preprocessed_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1,
            dataset_name=dataset_name)

        print('_________')
        test_solution(
            solution_name=solution_name,
            test_xses_series=test_xses_series,
            test_ys=test_ys,
            train_xses_series=train_xses_series,
            train_ys=train_ys,
            no_classes=no_classes,
            dataset_name=dataset_name,
            csv_results_path=plots_dir / f'{dataset_name}_{solution_name}_classification_results.csv')
