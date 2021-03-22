# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
from datetime import datetime

from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from transformingData import trajectory_slicing_linear
from transformingData import trajectory_slicing_sigmoid
from transformingData import trajectory_slicing_opposite
from testingResults import accuracyComparing
from examiningData import basicExamining
from loadingData import loadArff

TEST_ONLY_RF = True

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def test_different_classifiers(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title):
    np.random.seed(0)
    no_random_centers = 20

    mainplot_xs = []
    mainplot_ys_centroids_rf = []
    mainplot_ys_centroids_bp = []
    mainplot_ys_random_mean_rf = []
    mainplot_ys_random_mean_bp = []
    print(f"{plot_title}")

    mins, maxs = normalizing.get_mins_and_maxs(train_xses_series + test_xses_series)

    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    for no_centers in tqdm(range(3, 15)):
        centerss = [np.multiply(np.asarray([np.random.rand(input_size) for _1 in range(no_centers)]), maxs-mins)+mins
            for _ in range(no_random_centers)]

        # print(f"no_centers {no_centers}")
        # print("Calculating centers with clustering")
        centerss.append(cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers)[0])

        rf_accuracies = []
        bp_accuracies = []

        # print("Performing classification")
        for centers in centerss:

            train_xses_series_transformed = cmeans.transform(
                xses_series=train_xses_series,
                centers=centers)
            
            train_models = []

            for xs, y in zip(train_xses_series_transformed, train_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                train_models.append(mppi)
        
            test_xses_series_transformed = cmeans.transform(
                xses_series=test_xses_series,
                centers=centers)

            test_models = []

            for xs, y in zip(test_xses_series_transformed, test_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                test_models.append(mppi)
            
            # print("Calculating rf accuracy")
            rf_accuracy = accuracyComparing.get_accuracy(
                train_models=train_models,
                test_models=test_models,
                test_xs=test_xses_series_transformed,
                input_size=no_centers,
                no_classes=no_classes)

            # print("Calculating best_prediction accuracy")
            best_prediction_accuracy = accuracyComparing.get_accuracy(
                train_models=train_models,
                test_models=test_models,
                test_xs=test_xses_series_transformed,
                input_size=no_centers,
                no_classes=no_classes,
                classification_method='best_prediction')

            rf_accuracies.append(rf_accuracy)
            bp_accuracies.append(best_prediction_accuracy)

        mainplot_xs.append(no_centers)
        mainplot_ys_centroids_rf.append(rf_accuracies[no_random_centers])
        mainplot_ys_centroids_bp.append(bp_accuracies[no_random_centers])
        mainplot_ys_random_mean_rf.append(np.mean(rf_accuracies[:no_random_centers]))
        mainplot_ys_random_mean_bp.append(np.mean(bp_accuracies[:no_random_centers]))

    fig, ax = plt.subplots()
    ax.plot(mainplot_xs, mainplot_ys_centroids_rf, color='red', label='random forest')
    ax.plot(mainplot_xs, mainplot_ys_centroids_bp, color='blue', label='best prediction')
    ax.set(xlabel='number of centers', ylabel='classification accuracy', title=f'{plot_title} centroids algorithm')
    ax.grid()
    ax.legend()
    plt.savefig(plots_dir / f'{plot_title} centroids algorithm.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(mainplot_xs, mainplot_ys_random_mean_rf, color='red', label='random forest')
    ax.plot(mainplot_xs, mainplot_ys_random_mean_bp, color='blue', label='best prediction')
    ax.set(xlabel='number of centers', ylabel='classification accuracy', title=f'{plot_title} mean of {no_random_centers} random centers')
    ax.grid()
    ax.legend()
    plt.savefig(plots_dir / f'{plot_title} random centers.png')
    plt.close()


def perform_tests(data_loading_function, test_path, train_path, no_classes, input_size, derivative_order, plot_title):
    test_xses_series, test_ys = data_loading_function(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = data_loading_function(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    input_size = input_size * (derivative_order + 1)

    test_different_classifiers(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title)


if __name__ == "__main__":

    os.mkdir(plots_dir)

    # perform_tests(
    #     data_loading_function=loadArff.load_articulary_word_recognition,
    #     test_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.arff'),
    #     train_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff'),
    #     no_classes=25,
    #     input_size=9,
    #     derivative_order=0,
    #     plot_title='ArticularyWordRecognition different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_cricket,
        test_path=pathlib.Path('./data/Cricket/CRICKET_TEST.arff'),
        train_path=pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff'),
        no_classes=12,
        input_size=6,
        derivative_order=0,
        plot_title='Cricket different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_uwave,
        test_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff'),
        train_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff'),
        no_classes=12,
        input_size=3,
        derivative_order=0,
        plot_title='Uwave different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_basic_motions,
        test_path=pathlib.Path('./data/BasicMotions/BasicMotions_TEST.arff'),
        train_path=pathlib.Path('./data/BasicMotions/BasicMotions_TRAIN.arff'),
        no_classes=4,
        input_size=6,
        derivative_order=0,
        plot_title='BasicMotions different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_atrial_fibrilation,
        test_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TEST.arff'),
        train_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TRAIN.arff'),
        no_classes=3,
        input_size=2,
        derivative_order=0,
        plot_title='AtrialFibrillation different classifiers')

    # perform_tests(
    #     data_loading_function=loadArff.load_eigen_worms,
    #     test_path=pathlib.Path('./data/EigenWorms/EigenWorms_TEST.arff'),
    #     train_path=pathlib.Path('./data/EigenWorms/EigenWorms_TRAIN.arff'),
    #     no_classes=5,
    #     input_size=6,
    #     derivative_order=0,
    #     plot_title='EigenWorms different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_epilepsy,
        test_path=pathlib.Path('./data/Epilepsy/Epilepsy_TEST.arff'),
        train_path=pathlib.Path('./data/Epilepsy/Epilepsy_TRAIN.arff'),
        no_classes=4,
        input_size=3,
        derivative_order=0,
        plot_title='Epilepsy different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_ethanol_concentration,
        test_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TEST.arff'),
        train_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TRAIN.arff'),
        no_classes=4,
        input_size=3,
        derivative_order=0,
        plot_title='EthanolConcentration different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_ering,
        test_path=pathlib.Path('./data/ERing/ERing_TEST.arff'),
        train_path=pathlib.Path('./data/ERing/ERing_TRAIN.arff'),
        no_classes=6,
        input_size=4,
        derivative_order=0,
        plot_title='ERing different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_finger_movements,
        test_path=pathlib.Path('./data/FingerMovements/FingerMovements_TEST.arff'),
        train_path=pathlib.Path('./data/FingerMovements/FingerMovements_TRAIN.arff'),
        no_classes=2,
        input_size=28,
        derivative_order=0,
        plot_title='FingerMovements different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_hand_movement_direction,
        test_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TEST.arff'),
        train_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TRAIN.arff'),
        no_classes=4,
        input_size=10,
        derivative_order=0,
        plot_title='HandMovementDirection different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_handwriting,
        test_path=pathlib.Path('./data/Handwriting/Handwriting_TEST.arff'),
        train_path=pathlib.Path('./data/Handwriting/Handwriting_TRAIN.arff'),
        no_classes=26,
        input_size=3,
        derivative_order=0,
        plot_title='Handwriting different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_libras,
        test_path=pathlib.Path('./data/Libras/Libras_TEST.arff'),
        train_path=pathlib.Path('./data/Libras/Libras_TRAIN.arff'),
        no_classes=15,
        input_size=2,
        derivative_order=0,
        plot_title='Libras different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_natops,
        test_path=pathlib.Path('./data/NATOPS/NATOPS_TEST.arff'),
        train_path=pathlib.Path('./data/NATOPS/NATOPS_TRAIN.arff'),
        no_classes=6,
        input_size=24,
        derivative_order=0,
        plot_title='NATOPS different classifiers')

    # perform_tests(
    #     data_loading_function=loadArff.load_pen_digits,
    #     test_path=pathlib.Path('./data/PenDigits/PenDigits_TEST.arff'),
    #     train_path=pathlib.Path('./data/PenDigits/PenDigits_TRAIN.arff'),
    #     no_classes=10,
    #     input_size=2,
    #     derivative_order=0,
    #     plot_title='PenDigits different classifiers')

    # perform_tests(
    #     data_loading_function=loadArff.load_phoneme,
    #     test_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TEST.arff'),
    #     train_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TRAIN.arff'),
    #     no_classes=39,
    #     input_size=11,
    #     derivative_order=0,
    #     plot_title='PhonemeSpectra different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_racket_sports,
        test_path=pathlib.Path('./data/RacketSports/RacketSports_TEST.arff'),
        train_path=pathlib.Path('./data/RacketSports/RacketSports_TRAIN.arff'),
        no_classes=4,
        input_size=6,
        derivative_order=0,
        plot_title='RacketSports different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp1,
        test_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.arff'),
        no_classes=2,
        input_size=6,
        derivative_order=0,
        plot_title='SelfRegulationSCP1 different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp2,
        test_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.arff'),
        no_classes=2,
        input_size=7,
        derivative_order=0,
        plot_title='SelfRegulationSCP2 different classifiers')

    perform_tests(
        data_loading_function=loadArff.load_stand_walk_jump,
        test_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TEST.arff'),
        train_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TRAIN.arff'),
        no_classes=3,
        input_size=4,
        derivative_order=0,
        plot_title='StandWalkJump different classifiers')
