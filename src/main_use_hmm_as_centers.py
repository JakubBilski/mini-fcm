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


def test_fcm_set_centers(
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path):
    
    np.random.seed = 0

    print(f"{dataset_name}")

    ms = [2.0]

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(['dataset', 'no_classes'])
    csv_writer.writerow([dataset_name, no_classes])
    csv_writer.writerow(['m', 'no_centers', 'accuracy', 'centers'])

    for m in ms:
        print(f'm={m}')
        mainplot_xs = []
        mainplot_ys = []
        for no_centers in range(2, 11):
            print(f'\nno_centers={no_centers}')
            print(f'transforming with cmeans')

            centers, train_xses_series_transformed = cmeans.find_centers_and_transform(
                xses_series=train_xses_series,
                c=no_centers,
                m=m)
        
            test_xses_series_transformed = cmeans.transform(
                xses_series=test_xses_series,
                centers=centers)

            train_models = []

            print(f'learning train')
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

            print(f'classifying with best_prediction')
            accuracy = accuracyComparing.get_accuracy(
                train_models=train_models,
                test_models=test_models,
                test_xs=test_xses_series_transformed,
                input_size=no_centers,
                no_classes=no_classes,
                classification_method="best_prediction")
            print(f'accuracy: {accuracy}')            

            mainplot_xs.append(no_centers)
            mainplot_ys.append(accuracy)
            csv_writer.writerow([m, no_centers, accuracy] + centers)

        fig, ax = plt.subplots()
        ax.plot(mainplot_xs, mainplot_ys, color='blue')
        ax.set(xlabel='# centers', ylabel='classification accuracy', title=f'{dataset_name} decm m {m}')
        ax.grid()
        plt.savefig(plots_dir / f'{dataset_name} decm m {m}.png')
        plt.close()

    csv_results_file.close()


def create_hmms_for_classes(
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    no_states):

    train_xses_series_by_ys = [[] for _ in range(no_classes)]

    for xs, y in zip(train_xses_series, train_ys):
        train_xses_series_by_ys[int(y)].append(xs)

    hmm_by_ys = [HMM(no_states) for _ in range(no_classes)]

    for y in tqdm(range(no_classes)):
        hmm_by_ys[y].train(train_xses_series_by_ys[y], 100)
        hmm_by_ys[y].set_class(y)

    return hmm_by_ys


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


def generate_centers(
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path,
    no_centers):

    print("Learning centers with cmeans")
    centers, transformed_train_xses_series = cmeans.find_centers_and_transform(train_xses_series, no_centers)
    subtitles = [f'class {y}' for y in range(no_classes)]
    subtitles.append('cmeans')

    print("Learning hmms")
    hmms = create_hmms_for_classes(
        train_xses_series,
        train_ys,
        no_classes,
        dataset_name,
        no_centers
    )
    hmm_centers = [hmm.get_gauss_means() for hmm in hmms]

    print("Printing results")
    displaying.display_series_with_different_markers(
        train_xses_series,
        plots_dir / f'{dataset_name} centers.png',
        f'{dataset_name} centers',
        subtitles,
        [*hmm_centers, centers]
    )

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file, delimiter=';')
    csv_writer.writerow(['dataset', 'no_classes', 'no_centers'])
    csv_writer.writerow([dataset_name, no_classes, no_centers])
    csv_writer.writerow(['y', 'centers'])

    for y in range(no_classes):
        to_write = [f'hmm_{y}']
        to_write.extend(str(centers) for centers in hmm_centers[y])
        csv_writer.writerow(to_write)
    csv_writer.writerow(["cmeans"] + centers)
    csv_results_file.close()


def use_centers(
    train_xses_series,
    train_ys,
    test_xses_series,
    test_ys,
    csv_path):

    print("Reading csv")
    csv_file = open(csv_path, newline='')
    reader = csv.reader(csv_file, delimiter=';', quotechar='|')
    lines = [line for line in reader]
    dataset_name = lines[1][0]
    no_classes = int(lines[1][1])
    no_centers = int(lines[1][2])

    centers_by_ys = [[] for _ in range(no_classes)]

    for row in lines[3:]:
        class_no = row[0]
        if class_no == 'cmeans':
            continue
        class_no = int(class_no[4:])
        for i in range(no_centers):
            casted_to_list = [float(x) for x in row[i+1][1:-1].split(', ')]
            centers_by_ys[class_no].append(casted_to_list)

    train_xses_series_by_ys = [[] for y in range(no_classes)]

    for xs, y in zip(train_xses_series, train_ys):
        train_xses_series_by_ys[int(y)].append(xs)


    print("Transforming train xses according to their class")
    transformed_train_xses_series_by_ys = [[] for y in range(no_classes)]
    for y in range(no_classes):
        transformed_train_xses_series_by_ys[y] = cmeans.transform(train_xses_series_by_ys[y], centers_by_ys[y])
    
    print(f'Learning train models')
    train_models_by_ys = [[] for y in range(no_classes)]
    for y in tqdm(range(no_classes)):
        for xs in transformed_train_xses_series_by_ys[y]:
            model = DECognitiveMap(no_centers)
            model.train(xs)
            model.set_class(y)
            train_models_by_ys[y].append(model)
    
    test_models = []

    for y in test_ys:
        model = DECognitiveMap(no_centers)
        model.set_class(y)
        test_models.append(model)
    
    print("Example model:")
    print(train_models_by_ys[0][0].weights)

    print("Transforming test ys to all classes' centers")
    transformed_test_xses_series_by_ys = [[] for y in range(no_classes)]
    for y in range(no_classes):
        transformed_test_xses_series_by_ys[y] = cmeans.transform(test_xses_series, centers_by_ys[y])

    print(f'classifying with best_prediction')
    accuracy = accuracyComparing.get_accuracy_best_prediction_multicenter(
        train_models_by_ys=train_models_by_ys,
        test_models=test_models,
        test_xs_by_ys=transformed_test_xses_series_by_ys)
    print(f'accuracy: {accuracy}')


if __name__ == "__main__":

    os.mkdir(plots_dir)

    datasets = [
        ('ACSF1', 10),
        ('Adiac', 36),
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

    no_centers = 5
    mode_generate = True

    for dataset_name, no_classes in datasets:
        csv_path = f'plots\\picked\\centers_generated_with_hmm\\{dataset_name}_5_hmm_centers.csv'

        print(f"{dataset_name}")
        print(f"no_centers: {no_centers}")
        train_xses_series, train_ys, test_xses_series, test_ys = load_all_data(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            derivative_order=1)
        if mode_generate:
            generate_centers(
                train_xses_series=train_xses_series,
                train_ys=train_ys,
                no_classes=no_classes,
                dataset_name=dataset_name,
                csv_results_path=csv_path,
                no_centers=no_centers)
        else:
            use_centers(
                train_xses_series=train_xses_series,
                train_ys=train_ys,
                test_xses_series=test_xses_series,
                test_ys=test_ys,
                csv_path=csv_path)

