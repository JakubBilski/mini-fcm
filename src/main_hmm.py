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


def test_fcm(
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
        for no_centers in range(2, 4):
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

            print(f'learning test')
            for xs, y in tqdm(zip(test_xses_series_transformed, test_ys)):
                model = DECognitiveMap(no_centers)
                model.train(xs)
                # print(mppi.weights)
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


def test_hmm(
    test_xses_series,
    test_ys,
    train_xses_series,
    train_ys,
    no_classes,
    dataset_name,
    csv_results_path):

    mainplot_xs = []
    mainplot_ys = []
    print(f"{dataset_name}")

    csv_results_file = open(csv_results_path, 'w', newline='')
    csv_writer = csv.writer(csv_results_file)
    csv_writer.writerow(['dataset', 'no_classes'])
    csv_writer.writerow([dataset_name, no_classes])
    csv_writer.writerow(['no_states', 'accuracy'])

    for no_states in tqdm(range(2, 4)):
        hmm_train_models = []
        hmm_error_occured = False

        for xs, y in zip(train_xses_series, train_ys):
            hmm = HMM(no_states)
            try:
                hmm.train(xs, 100)
            except:
                hmm_error_occured = True
                break
            hmm.set_class(y)
            hmm_train_models.append(hmm)

        if not hmm_error_occured:
            try:
                accuracy = accuracyComparing.get_accuracy_hmm(
                    train_models=hmm_train_models,
                    test_xs=test_xses_series,
                    test_classes=test_ys,
                    input_size=no_states,
                    no_classes=no_classes
                )
            except:
                hmm_error_occured = True
        
        if hmm_error_occured:
            print(f"hmm error occured for no_states {no_states}")
            break
        else:
            mainplot_xs.append(no_states)
            mainplot_ys.append(accuracy)
            csv_writer.writerow([no_states, accuracy])
    
    fig, ax = plt.subplots()
    ax.plot(mainplot_xs, mainplot_ys, color='blue')
    ax.set(xlabel='# hidden states', ylabel='classification accuracy', title=f'{dataset_name} hmm')
    ax.grid()
    plt.savefig(plots_dir / f'{dataset_name} hmm.png')
    plt.close()

    csv_results_file.close()


def perform_tests(
    data_loading_function,
    test_path,
    train_path,
    no_classes,
    derivative_order,
    dataset_name,
    fcm_csv_path,
    hmm_csv_path,
    testing_fcm=False,
    testing_hmm=False):

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

    if testing_fcm:
        test_fcm(
            test_xses_series,
            test_ys,
            train_xses_series,
            train_ys,
            no_classes,
            dataset_name,
            fcm_csv_path)
    
    if testing_hmm:
        test_hmm(
            test_xses_series,
            test_ys,
            train_xses_series,
            train_ys,
            no_classes,
            dataset_name,
            hmm_csv_path
        )


if __name__ == "__main__":

    os.mkdir(plots_dir)

    testing_fcm = True
    testing_hmm = False

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

    datasets = [datasets[1]]

    for dataset_name, no_classes in datasets:
        perform_tests(
            data_loading_function=loadSktime.load_sktime,
            test_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TEST.ts'),
            train_path=pathlib.Path(f'./data/Univariate/{dataset_name}/{dataset_name}_TRAIN.ts'),
            no_classes=no_classes,
            derivative_order=1,
            dataset_name=dataset_name,
            fcm_csv_path=plots_dir / f'{dataset_name}_fcm_csv_path.csv',
            hmm_csv_path=plots_dir / f'{dataset_name}_hmm_csv_path.csv',
            testing_fcm=testing_fcm,
            testing_hmm=testing_hmm)

