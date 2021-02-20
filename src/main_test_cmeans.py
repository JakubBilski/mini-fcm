# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

from cognitiveMaps import mppiCheckpoints
from cognitiveMaps import basicComparing
from cognitiveMaps import rfClassifier
from cognitiveMaps import svmClassifier
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from transformingData import cmeans
from transformingData import derivatives
from transformingData import normalizing
from transformingData import trajectory_slicing_linear
from transformingData import trajectory_slicing_sigmoid
from transformingData import trajectory_slicing_opposite
from loadingData import loadArff
from datetime import datetime

TEST_ONLY_RF = True

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

def compare_solutions(train_models, test_models, test_xs, input_size, no_classes):
    mistakes = 0
    if not TEST_ONLY_RF:
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit, best_cost = basicComparing.nn_weights(train_models_without_same, test_model, input_size, input_size)
            if best_fit.get_class() != test_model.get_class():
                good_predictions = [m for m in train_models_without_same if m.get_class()==test_model.get_class()]
                best_correct_fit, best_correct_cost = basicComparing.nn_weights(good_predictions, test_model, input_size, input_size)
                mistakes += 1
            acc = 1-mistakes/len(test_models)
            print(f"nn_weights accuracy: {len(test_models)-mistakes}/{len(test_models)} ({acc})")


    if not TEST_ONLY_RF:
        if test_xs:
            mistakes = 0
            for test_model, xs in zip(test_models, test_xs):
                train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
                best_fit = basicComparing.nn_convergence(train_models_without_same, test_model, xs[0])
                if best_fit.get_class() != test_model.get_class():
                    mistakes += 1
            print(f"nn_convergence accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    if not TEST_ONLY_RF:
        mistakes = 0
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            fit_class, _ = basicComparing.best_mse_sum(train_models_without_same, test_model, no_classes)
            if fit_class != test_model.get_class():
                mistakes += 1
        print(f"best_mse_sum accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    mistakes = 0
    cfr = rfClassifier.RFClassifier(
        [tm.weights for tm in train_models],
        [tm.class_name for tm in train_models]
        )
    for test_model in test_models:
        fit_class = cfr.predict(test_model.weights)
        if fit_class != test_model.get_class():
            mistakes += 1
    # print(f"rfClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    rf_accuracy = 1-mistakes/len(test_models)

    if not TEST_ONLY_RF:
        mistakes = 0
        cfr = svmClassifier.SVMClassifier(
            [tm.weights for tm in train_models],
            [tm.class_name for tm in train_models]
            )
        for test_model in test_models:
            fit_class = cfr.predict(test_model.weights)
            if fit_class != test_model.get_class():
                mistakes += 1
        # print(f"SVMClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    return rf_accuracy


def generate_mppi_checkpoints(derivative_order, no_centers):
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/mppi/{no_centers}_{derivative_order}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    centers, xses_series = cmeans.find_centers_and_transform(xses_series, c=no_centers)

    mppiCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        input_size=no_centers,
        cmeans_centers=centers)

    input_path = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/mppi/{no_centers}_{derivative_order}/test/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    xses_series = cmeans.transform(xses_series, centers=centers)

    mppiCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        input_size=no_centers,
        cmeans_centers=centers)


def test_different_cmeans(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title):
    np.random.seed = 0
    no_random_centers = 1
    print(f"{plot_title}")
    # print("Calculating results for bare input")

    mins, maxs = normalizing.get_mins_and_maxs(train_xses_series + test_xses_series)

    train_xses_series_transformed = normalizing.transform(
        xses_series=train_xses_series,
        mins = mins,
        maxs = maxs)
    
    train_models = []

    for xs, y in zip(train_xses_series_transformed, train_ys):
        mppi = MppiCognitiveMap(input_size)
        mppi.train(xs)
        mppi.set_class(y)
        train_models.append(mppi)

    test_xses_series_transformed = normalizing.transform(
        xses_series=test_xses_series,
        mins = mins,
        maxs = maxs)

    test_models = []

    for xs, y in zip(test_xses_series_transformed, test_ys):
        mppi = MppiCognitiveMap(input_size)
        mppi.train(xs)
        mppi.set_class(y)
        test_models.append(mppi)

    bare_rf_accuracy = compare_solutions(
        train_models=train_models,
        test_models=test_models,
        test_xs=test_xses_series_transformed,
        input_size=input_size,
        no_classes=no_classes)

    bare_err = sum([mppi.get_error(xs) for mppi, xs in zip(train_models, train_xses_series_transformed)])/len(train_models)
    bare_volatility = basicComparing.get_volatility_taxicab(xs)

    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    for no_centers in tqdm(range(3, 4)):
        random_centerss = [np.multiply(np.asarray([np.random.rand(input_size) for _1 in range(no_centers)]), maxs-mins)+mins
            for _ in range(no_random_centers)]

        # print(f"no_centers {no_centers}")
        # print("Calculating centers with clustering")
        random_centerss.append(cmeans.find_centers_and_transform(
            xses_series=train_xses_series,
            c=no_centers)[0])
        # print("Calculating other custom centers")
        sliced_train_xses_series = trajectory_slicing_linear.transform(train_xses_series)
        random_centerss.append(cmeans.find_centers_in_first_and_transform_second(
            first_series=sliced_train_xses_series,
            second_series=train_xses_series,
            c=no_centers)[0])
        sliced_sigmoid_train_xses_series = trajectory_slicing_sigmoid.transform(train_xses_series)
        random_centerss.append(cmeans.find_centers_in_first_and_transform_second(
            first_series=sliced_sigmoid_train_xses_series,
            second_series=train_xses_series,
            c=no_centers)[0])
        sliced_sigmoid_train_xses_series = trajectory_slicing_opposite.transform(train_xses_series)
        random_centerss.append(cmeans.find_centers_in_first_and_transform_second(
            first_series=sliced_sigmoid_train_xses_series,
            second_series=train_xses_series,
            c=no_centers)[0])

        plot_xs, plot_ys = [], []
        plot2_xs, plot2_ys = [], []
        plot3_xs, plot3_ys = [], []

        # print("Performing classification")
        for random_centers in random_centerss:

            train_xses_series_transformed = cmeans.transform(
                xses_series=train_xses_series,
                centers=random_centers)
            
            train_models = []

            for xs, y in zip(train_xses_series_transformed, train_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                train_models.append(mppi)
        
            test_xses_series_transformed = cmeans.transform(
                xses_series=test_xses_series,
                centers=random_centers)

            test_models = []

            for xs, y in zip(test_xses_series_transformed, test_ys):
                mppi = MppiCognitiveMap(no_centers)
                mppi.train(xs)
                mppi.set_class(y)
                test_models.append(mppi)

            rf_accuracy = compare_solutions(
                train_models=train_models,
                test_models=test_models,
                test_xs=test_xses_series_transformed,
                input_size=no_centers,
                no_classes=no_classes)

            err = sum([mppi.get_error(xs) for mppi, xs in zip(train_models, train_xses_series_transformed)])/len(train_models)
            # print(f'Prediction error: {err}')
            volatility = basicComparing.get_volatility_taxicab(xs)

            plot_xs.append(err)
            plot_ys.append(rf_accuracy)

            plot2_xs.append(volatility / err)
            plot2_ys.append(rf_accuracy)
            
            plot3_xs.append(volatility)
            plot3_ys.append(err)


        fig, ax = plt.subplots()
        ax.plot(plot_xs[:-4], plot_ys[:-4], 'bo')
        ax.set(xlabel='prediction error', ylabel='classification accuracy (rf)',
        title=f'{plot_title}, no_centers {no_centers}')
        ax.grid()
        ax.scatter(plot_xs[no_random_centers], plot_ys[no_random_centers], color='red')
        ax.scatter(plot_xs[no_random_centers+1], plot_ys[no_random_centers+1], color='green')
        ax.scatter(plot_xs[no_random_centers+2], plot_ys[no_random_centers+2], color='orange')
        ax.scatter(plot_xs[no_random_centers+3], plot_ys[no_random_centers+3], color='purple')
        ax.scatter(bare_err, bare_rf_accuracy, color='brown')
        plt.savefig(plots_dir / f'{plot_title} no_centers {no_centers} pred vs rf.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(plot2_xs[:-4], plot2_ys[:-4], 'bo')
        ax.set(xlabel='volatility / predition error', ylabel='classification accuracy (rf)',
               title=f'{plot_title}, no_centers {no_centers}')
        ax.grid()
        ax.scatter(plot2_xs[no_random_centers], plot2_ys[no_random_centers], color='red')
        ax.scatter(plot2_xs[no_random_centers+1], plot2_ys[no_random_centers+1], color='green')
        ax.scatter(plot2_xs[no_random_centers+2], plot2_ys[no_random_centers+2], color='orange')
        ax.scatter(plot2_xs[no_random_centers+3], plot2_ys[no_random_centers+3], color='purple')
        ax.scatter(bare_volatility / bare_err, bare_rf_accuracy, color='brown')
        plt.savefig(plots_dir / f'{plot_title} no_centers {no_centers} vot div pred vs rf.png')
        plt.close()

    
        fig, ax = plt.subplots()
        ax.plot(plot3_xs[:-4], plot3_ys[:-4], 'bo')
        ax.set(xlabel='volatility', ylabel='prediction error',
               title=f'{plot_title}, no_centers {no_centers}')
        ax.scatter(plot3_xs[no_random_centers], plot3_ys[no_random_centers], color='red')
        ax.scatter(plot3_xs[no_random_centers+1], plot3_ys[no_random_centers+1], color='green')
        ax.scatter(plot3_xs[no_random_centers+2], plot3_ys[no_random_centers+2], color='orange')
        ax.scatter(plot3_xs[no_random_centers+3], plot3_ys[no_random_centers+3], color='purple')
        ax.scatter(bare_volatility, bare_err, color='brown')
        ax.grid()
        plt.savefig(plots_dir / f'{plot_title} no_centers {no_centers} vot vs pred.png')
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

    test_different_cmeans(test_xses_series, test_ys, train_xses_series, train_ys, no_classes, input_size, plot_title)



if __name__ == "__main__":

    os.mkdir(plots_dir)

    # perform_tests(
    #     data_loading_function=loadArff.load_articulary_word_recognition,
    #     test_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.arff'),
    #     train_path=pathlib.Path('./data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff'),
    #     no_classes=25,
    #     input_size=9,
    #     derivative_order=0,
    #     plot_title='ArticularyWordRecognition different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_cricket,
    #     test_path=pathlib.Path('./data/Cricket/CRICKET_TEST.arff'),
    #     train_path=pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff'),
    #     no_classes=12,
    #     input_size=6,
    #     derivative_order=0,
    #     plot_title='Cricket different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_uwave,
    #     test_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff'),
    #     train_path=pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.arff'),
    #     no_classes=12,
    #     input_size=3,
    #     derivative_order=0,
    #     plot_title='Uwave different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_basic_motions,
    #     test_path=pathlib.Path('./data/BasicMotions/BasicMotions_TEST.arff'),
    #     train_path=pathlib.Path('./data/BasicMotions/BasicMotions_TRAIN.arff'),
    #     no_classes=4,
    #     input_size=6,
    #     derivative_order=0,
    #     plot_title='BasicMotions different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_atrial_fibrilation,
    #     test_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TEST.arff'),
    #     train_path=pathlib.Path('./data/AtrialFibrillation/AtrialFibrillation_TRAIN.arff'),
    #     no_classes=3,
    #     input_size=2,
    #     derivative_order=0,
    #     plot_title='AtrialFibrillation different centers')

    # # perform_tests(
    # #     data_loading_function=loadArff.load_eigen_worms,
    # #     test_path=pathlib.Path('./data/EigenWorms/EigenWorms_TEST.arff'),
    # #     train_path=pathlib.Path('./data/EigenWorms/EigenWorms_TRAIN.arff'),
    # #     no_classes=5,
    # #     input_size=6,
    # #     derivative_order=0,
    # #     plot_title='EigenWorms different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_epilepsy,
    #     test_path=pathlib.Path('./data/Epilepsy/Epilepsy_TEST.arff'),
    #     train_path=pathlib.Path('./data/Epilepsy/Epilepsy_TRAIN.arff'),
    #     no_classes=4,
    #     input_size=3,
    #     derivative_order=0,
    #     plot_title='Epilepsy different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_ethanol_concentration,
    #     test_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TEST.arff'),
    #     train_path=pathlib.Path('./data/EthanolConcentration/EthanolConcentration_TRAIN.arff'),
    #     no_classes=4,
    #     input_size=3,
    #     derivative_order=0,
    #     plot_title='EthanolConcentration different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_ering,
    #     test_path=pathlib.Path('./data/ERing/ERing_TEST.arff'),
    #     train_path=pathlib.Path('./data/ERing/ERing_TRAIN.arff'),
    #     no_classes=6,
    #     input_size=4,
    #     derivative_order=0,
    #     plot_title='ERing different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_finger_movements,
    #     test_path=pathlib.Path('./data/FingerMovements/FingerMovements_TEST.arff'),
    #     train_path=pathlib.Path('./data/FingerMovements/FingerMovements_TRAIN.arff'),
    #     no_classes=2,
    #     input_size=28,
    #     derivative_order=0,
    #     plot_title='FingerMovements different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_hand_movement_direction,
    #     test_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TEST.arff'),
    #     train_path=pathlib.Path('./data/HandMovementDirection/HandMovementDirection_TRAIN.arff'),
    #     no_classes=4,
    #     input_size=10,
    #     derivative_order=0,
    #     plot_title='HandMovementDirection different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_handwriting,
    #     test_path=pathlib.Path('./data/Handwriting/Handwriting_TEST.arff'),
    #     train_path=pathlib.Path('./data/Handwriting/Handwriting_TRAIN.arff'),
    #     no_classes=26,
    #     input_size=3,
    #     derivative_order=0,
    #     plot_title='Handwriting different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_libras,
    #     test_path=pathlib.Path('./data/Libras/Libras_TEST.arff'),
    #     train_path=pathlib.Path('./data/Libras/Libras_TRAIN.arff'),
    #     no_classes=15,
    #     input_size=2,
    #     derivative_order=0,
    #     plot_title='Libras different centers')

    # perform_tests(
    #     data_loading_function=loadArff.load_natops,
    #     test_path=pathlib.Path('./data/NATOPS/NATOPS_TEST.arff'),
    #     train_path=pathlib.Path('./data/NATOPS/NATOPS_TRAIN.arff'),
    #     no_classes=6,
    #     input_size=24,
    #     derivative_order=0,
    #     plot_title='NATOPS different centers')

    # # perform_tests(
    # #     data_loading_function=loadArff.load_pen_digits,
    # #     test_path=pathlib.Path('./data/PenDigits/PenDigits_TEST.arff'),
    # #     train_path=pathlib.Path('./data/PenDigits/PenDigits_TRAIN.arff'),
    # #     no_classes=10,
    # #     input_size=2,
    # #     derivative_order=0,
    # #     plot_title='PenDigits different centers')

    perform_tests(
        data_loading_function=loadArff.load_phoneme,
        test_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TEST.arff'),
        train_path=pathlib.Path('./data/PhonemeSpectra/PhonemeSpectra_TRAIN.arff'),
        no_classes=39,
        input_size=11,
        derivative_order=0,
        plot_title='PhonemeSpectra different centers')

    perform_tests(
        data_loading_function=loadArff.load_racket_sports,
        test_path=pathlib.Path('./data/RacketSports/RacketSports_TEST.arff'),
        train_path=pathlib.Path('./data/RacketSports/RacketSports_TRAIN.arff'),
        no_classes=4,
        input_size=6,
        derivative_order=0,
        plot_title='RacketSports different centers')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp1,
        test_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.arff'),
        no_classes=2,
        input_size=6,
        derivative_order=0,
        plot_title='SelfRegulationSCP1 different centers')

    perform_tests(
        data_loading_function=loadArff.load_self_regulation_scp2,
        test_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TEST.arff'),
        train_path=pathlib.Path('./data/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.arff'),
        no_classes=2,
        input_size=7,
        derivative_order=0,
        plot_title='SelfRegulationSCP2 different centers')

    perform_tests(
        data_loading_function=loadArff.load_spoken_arabic_digits,
        test_path=pathlib.Path('./data/SpokenArabicDigits/SpokenArabicDigits_TEST.arff'),
        train_path=pathlib.Path('./data/SpokenArabicDigits/SpokenArabicDigits_TRAIN.arff'),
        no_classes=10,
        input_size=13,
        derivative_order=0,
        plot_title='SpokenArabicDigits different centers')

    perform_tests(
        data_loading_function=loadArff.load_stand_walk_jump,
        test_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TEST.arff'),
        train_path=pathlib.Path('./data/StandWalkJump/StandWalkJump_TRAIN.arff'),
        no_classes=3,
        input_size=4,
        derivative_order=0,
        plot_title='StandWalkJump different centers')
