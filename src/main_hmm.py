# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
from datetime import datetime

from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from cognitiveMaps.hmm import HMM
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from testingResults import accuracyComparing
from examiningData import basicExamining
from loadingData import loadArff

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')


def test_hmm_against_fcm(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title):
    np.random.seed = 0

    mainplot_xs = []
    mainplot_fcm = []
    mainplot_hmm = []
    print(f"{plot_title}")

    mins, maxs = normalizing.get_mins_and_maxs(train_xses_series + test_xses_series)

    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    for no_centers in tqdm(range(3, 15)):
        fcm_centers, train_xses_series_transformed = cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers)
        
        fcm_train_models = []

        for xs, y in zip(train_xses_series_transformed, train_ys):
            mppi = MppiCognitiveMap(no_centers)
            mppi.train(xs)
            mppi.set_class(y)
            fcm_train_models.append(mppi)
    
        test_xses_series_transformed = cmeans.transform(
            xses_series=test_xses_series,
            centers=fcm_centers)

        fcm_test_models = []

        for xs, y in zip(test_xses_series_transformed, test_ys):
            mppi = MppiCognitiveMap(no_centers)
            mppi.train(xs)
            mppi.set_class(y)
            fcm_test_models.append(mppi)
        
        fcm_accuracy = accuracyComparing.get_accuracy(
            train_models=fcm_train_models,
            test_models=fcm_test_models,
            test_xs=test_xses_series_transformed,
            input_size=no_centers,
            no_classes=no_classes)

        hmm_train_models = []

        for xs, y in zip(train_xses_series, train_ys):
            hmm = HMM(no_centers)
            hmm.train(xs)
            hmm.set_class(y)
            hmm_train_models.append(hmm)

        hmm_accuracy = accuracyComparing.get_accuracy_hmm(
            train_models=hmm_train_models,
            test_xs=test_xses_series,
            test_classes=test_ys,
            input_size=no_centers,
            no_classes=no_classes
        )

        mainplot_xs.append(no_centers)
        mainplot_fcm.append(fcm_accuracy)
        mainplot_hmm.append(hmm_accuracy)

    fig, ax = plt.subplots()
    ax.plot(mainplot_xs, mainplot_fcm, color='red', label='fcm (centroids) random forest')
    ax.plot(mainplot_xs, mainplot_hmm, color='blue', label='hmm best prediction')
    ax.set(xlabel='# centers / # states', ylabel='classification accuracy', title=f'{plot_title}')
    ax.grid()
    ax.legend()
    plt.savefig(plots_dir / f'{plot_title}.png')
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

    test_hmm_against_fcm(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title)


if __name__ == "__main__":

    os.mkdir(plots_dir)

    # perform_tests(
    #     data_loading_function=loadArff.load_articulary_word_recognition,
    #     test_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.arff'),
    #     train_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff'),
    #     no_classes=25,
    #     input_size=9,
    #     derivative_order=0,
    #     plot_title='ArticularyWordRecognition fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_cricket,
        test_path=pathlib.Path('./data/Cricket/CRICKET_TEST.arff'),
        train_path=pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff'),
        no_classes=12,
        input_size=6,
        derivative_order=0,
        plot_title='Cricket fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_uwave,
        test_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff'),
        train_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff'),
        no_classes=12,
        input_size=3,
        derivative_order=0,
        plot_title='Uwave fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_basic_motions,
        test_path=pathlib.Path('./data/BasicMotions/BasicMotions_TEST.arff'),
        train_path=pathlib.Path('./data/BasicMotions/BasicMotions_TRAIN.arff'),
        no_classes=4,
        input_size=6,
        derivative_order=0,
        plot_title='BasicMotions fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_atrial_fibrilation,
        test_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TEST.arff'),
        train_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TRAIN.arff'),
        no_classes=3,
        input_size=2,
        derivative_order=0,
        plot_title='AtrialFibrillation fcm vs hmm classification')

    # perform_tests(
    #     data_loading_function=loadArff.load_eigen_worms,
    #     test_path=pathlib.Path('./data/EigenWorms/EigenWorms_TEST.arff'),
    #     train_path=pathlib.Path('./data/EigenWorms/EigenWorms_TRAIN.arff'),
    #     no_classes=5,
    #     input_size=6,
    #     derivative_order=0,
    #     plot_title='EigenWorms fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_epilepsy,
        test_path=pathlib.Path('./data/Epilepsy/Epilepsy_TEST.arff'),
        train_path=pathlib.Path('./data/Epilepsy/Epilepsy_TRAIN.arff'),
        no_classes=4,
        input_size=3,
        derivative_order=0,
        plot_title='Epilepsy fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_ethanol_concentration,
        test_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TEST.arff'),
        train_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TRAIN.arff'),
        no_classes=4,
        input_size=3,
        derivative_order=0,
        plot_title='EthanolConcentration fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_ering,
        test_path=pathlib.Path('./data/ERing/ERing_TEST.arff'),
        train_path=pathlib.Path('./data/ERing/ERing_TRAIN.arff'),
        no_classes=6,
        input_size=4,
        derivative_order=0,
        plot_title='ERing fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_finger_movements,
        test_path=pathlib.Path('./data/FingerMovements/FingerMovements_TEST.arff'),
        train_path=pathlib.Path('./data/FingerMovements/FingerMovements_TRAIN.arff'),
        no_classes=2,
        input_size=28,
        derivative_order=0,
        plot_title='FingerMovements fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_hand_movement_direction,
        test_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TEST.arff'),
        train_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TRAIN.arff'),
        no_classes=4,
        input_size=10,
        derivative_order=0,
        plot_title='HandMovementDirection fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_handwriting,
        test_path=pathlib.Path('./data/Handwriting/Handwriting_TEST.arff'),
        train_path=pathlib.Path('./data/Handwriting/Handwriting_TRAIN.arff'),
        no_classes=26,
        input_size=3,
        derivative_order=0,
        plot_title='Handwriting fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_libras,
        test_path=pathlib.Path('./data/Libras/Libras_TEST.arff'),
        train_path=pathlib.Path('./data/Libras/Libras_TRAIN.arff'),
        no_classes=15,
        input_size=2,
        derivative_order=0,
        plot_title='Libras fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_natops,
        test_path=pathlib.Path('./data/NATOPS/NATOPS_TEST.arff'),
        train_path=pathlib.Path('./data/NATOPS/NATOPS_TRAIN.arff'),
        no_classes=6,
        input_size=24,
        derivative_order=0,
        plot_title='NATOPS fcm vs hmm classification')

    # perform_tests(
    #     data_loading_function=loadArff.load_pen_digits,
    #     test_path=pathlib.Path('./data/PenDigits/PenDigits_TEST.arff'),
    #     train_path=pathlib.Path('./data/PenDigits/PenDigits_TRAIN.arff'),
    #     no_classes=10,
    #     input_size=2,
    #     derivative_order=0,
    #     plot_title='PenDigits fcm vs hmm classification')

    # perform_tests(
    #     data_loading_function=loadArff.load_phoneme,
    #     test_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TEST.arff'),
    #     train_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TRAIN.arff'),
    #     no_classes=39,
    #     input_size=11,
    #     derivative_order=0,
    #     plot_title='PhonemeSpectra fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_racket_sports,
        test_path=pathlib.Path('./data/RacketSports/RacketSports_TEST.arff'),
        train_path=pathlib.Path('./data/RacketSports/RacketSports_TRAIN.arff'),
        no_classes=4,
        input_size=6,
        derivative_order=0,
        plot_title='RacketSports fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp1,
        test_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.arff'),
        no_classes=2,
        input_size=6,
        derivative_order=0,
        plot_title='SelfRegulationSCP1 fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp2,
        test_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.arff'),
        no_classes=2,
        input_size=7,
        derivative_order=0,
        plot_title='SelfRegulationSCP2 fcm vs hmm classification')

    perform_tests(
        data_loading_function=loadArff.load_stand_walk_jump,
        test_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TEST.arff'),
        train_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TRAIN.arff'),
        no_classes=3,
        input_size=4,
        derivative_order=0,
        plot_title='StandWalkJump fcm vs hmm classification')
