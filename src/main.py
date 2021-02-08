# flake8: noqa
from tqdm import tqdm
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

from cognitiveMaps import ecmCheckpoints
from cognitiveMaps import fcmCheckpoints
from cognitiveMaps import displaying
from cognitiveMaps import basicComparing
from cognitiveMaps import rfClassifier
from cognitiveMaps import svmClassifier
from transformingData import cmeans
from transformingData import derivatives
from cognitiveMaps.mppiCognitiveMap import MppiCognitiveMap
from loadingData import loadArff


def compare_solutions(train_models, test_models, test_xs, test_ys, input_size, extend_size, no_classes):
    mistakes = 0
    for test_model in test_models:
        train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
        best_fit, best_cost = basicComparing.nn_weights(train_models_without_same, test_model, input_size+extend_size, input_size)
        if best_fit.get_class() != test_model.get_class():
            good_predictions = [m for m in train_models_without_same if m.get_class()==test_model.get_class()]
            best_correct_fit, best_correct_cost = basicComparing.nn_weights(good_predictions, test_model, input_size+extend_size, input_size)
            # displaying.draw_cognitive_maps(
            #     [
            #         best_fit.weights,
            #         test_model.weights,
            #         best_correct_fit.weights
            #     ],
            #     [
            #         f"best fit (class {best_fit.get_class()}) (cost {best_cost})",
            #         f"test model (class {test_model.get_class()})",
            #         f"best correct fit (cost {best_correct_cost})"
            #     ]
            # )
            # print(f"{best_fit.get_class()} should be {test_model.get_class()}")
            # print(f"{best_fit} won")
            mistakes += 1
    acc = 1-mistakes/len(test_models)
    print(f"nn_weights accuracy: {len(test_models)-mistakes}/{len(test_models)} ({acc})")


    if extend_size > 0:
        mistakes = 0
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit = basicComparing.nn_weights_and_start_values(train_models_without_same, test_model, input_size, extend_size)
            if best_fit.get_class() != test_model.get_class():
                mistakes += 1
        print(f"nn_weights_and_start_values accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

    # mistakes = 0
    # for test_model, xs in tqdm(zip(test_models, test_xs)):
    #     best_fit = basicComparing.best_prediction(train_models, xs)
    #     if best_fit.get_class() != test_model.get_class():
    #         mistakes += 1
    # print(f"best_prediction accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    if test_xs:
        mistakes = 0
        for test_model, xs in zip(test_models, test_xs):
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit = basicComparing.nn_convergence(train_models_without_same, test_model, xs[0])
            if best_fit.get_class() != test_model.get_class():
                mistakes += 1
        print(f"nn_convergence accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")

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
    print(f"rfClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    rf_accuracy = 1-mistakes/len(test_models)

    mistakes = 0
    cfr = svmClassifier.SVMClassifier(
        [tm.weights for tm in train_models],
        [tm.class_name for tm in train_models]
        )
    for test_model in test_models:
        fit_class = cfr.predict(test_model.weights)
        if fit_class != test_model.get_class():
            mistakes += 1
    print(f"SVMClassifier accuracy: {len(test_models)-mistakes}/{len(test_models)} ({1-mistakes/len(test_models)})")
    return rf_accuracy



def generate_ecm_checkpoints():
    extended_size = 1
    learning_rate = 0.002
    steps = 2
    input_size = 6
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/ecm/{extended_size}_{learning_rate}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    ecmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size,
        extended_size)


def generate_fcm_checkpoints():
    learning_rate = 0.002
    steps = 10
    input_size = 6
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/fcm/{learning_rate}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    fcmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size)


def generate_fcm_cmeans_checkpoints(learning_rate, derivative_order, no_centers, steps, save_step=1):
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/fcm_cmeans/{no_centers}_{learning_rate}_{derivative_order}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    centers, xses_series = cmeans.find_centers_and_transform(xses_series, c=no_centers)

    fcmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size=no_centers,
        save_step=save_step,
        cmeans_centers=centers)
    input_path = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/fcm_cmeans/{no_centers}_{learning_rate}_{derivative_order}/test/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    xses_series = cmeans.transform(xses_series, centers=centers)
    fcmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size=no_centers,
        save_step=save_step,
        cmeans_centers=centers)

def generate_ecm_cmeans_checkpoints(learning_rate, derivative_order, no_centers, steps, extended_size, save_step=1):
    input_path = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/ecm_cmeans/{no_centers}_{learning_rate}_{derivative_order}_{extended_size}/train/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    centers, xses_series = cmeans.find_centers_and_transform(xses_series, c=no_centers)

    ecmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size=no_centers,
        extended_size=extended_size,
        save_step=save_step,
        cmeans_centers=centers)

    input_path = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    output_path = pathlib.Path(f'./checkpoints/Cricket/ecm_cmeans/{no_centers}_{learning_rate}_{derivative_order}_{extended_size}/test/')
    # os.mkdir(output_path)
    os.mkdir(output_path)
    xses_series, ys = loadArff.load_cricket_normalized(input_path)
    if derivative_order > 0:
        xses_series = derivatives.transform(xses_series, derivative_order)
    xses_series = cmeans.transform(xses_series, centers=centers)
    ecmCheckpoints.create_checkpoints(
        xses_series,
        ys,
        output_path,
        learning_rate,
        steps,
        input_size=no_centers,
        extended_size=extended_size,
        save_step=save_step,
        cmeans_centers=centers)


if __name__ == "__main__":

    # for no_centers in range(7,10):
    #     for derivative_order in [0,1,2]:
    #         for extended_size in [2, no_centers//2, no_centers]:
    #             os.mkdir(pathlib.Path(f'./checkpoints/Cricket/ecm_cmeans/{no_centers}_{0.002}_{derivative_order}_{extended_size}/'))

    # for no_centers in range(2,50):
    #     for derivative_order in [0]:
    #         os.mkdir(pathlib.Path(f'./checkpoints/Cricket/fcm_cmeans/{no_centers}_{0.002}_{derivative_order}/'))


    # for no_centers in range(2,50):
    #     for derivative_order in [0]:
    #         print(f"Learning no_centers {no_centers}, derivative_order {derivative_order}")
    #         generate_fcm_cmeans_checkpoints(
    #             no_centers=no_centers,
    #             derivative_order=derivative_order,
    #             steps=200,
    #             learning_rate=0.002
    #         )


    # print(f"Learning no_centers {4}, derivative_order {0}")
    # generate_fcm_cmeans_checkpoints(
    #     no_centers=4,
    #     derivative_order=0,
    #     steps=10,
    #     learning_rate=0.002,
    #     save_step=10
    # )

    # # solution comparison for ecms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/test')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_training_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_training_paths = ecmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_training_paths[0].points)):
    #     print(f"Step {step}")
    #     train_models = [tp.points[step] for tp in train_training_paths]
    #     test_models = [tp.points[step] for tp in test_training_paths]
    #     compare_solutions(test_models, test_models, xses_series, None, 6, 3, 12)

    # # examine grouping factor based on k-means
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002_big_free_seed/test')
    # train_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_paths = ecmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_paths[0].points)):
    #     train_models = [train_path.points[step] for train_path in train_paths]
    #     test_models = [test_path.points[step] for test_path in test_paths]
    #     print(f"step {step}")
    #     print(f"train grouping factor: {comparing.get_grouping_factor(train_models, 6, 3, 12)}")
    #     print(f"test grouping factor: {comparing.get_grouping_factor(test_models, 6, 3, 12)}")
    #     print(f"all grouping factor: {comparing.get_grouping_factor(test_models+train_models, 6, 3, 12)}")

    # # examine error of ecms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/ecm/Cricket/3_0.002/train')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_paths = ecmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # for train_path in train_paths:
    #     print(f"New train path (class {train_path.class_name})")
    #     if train_path.class_name != ys[train_path.input_data_index]:
    #         print("Mismatch!")
    #     if train_path.cmeans_centers:
    #         xses_series = cmeans.transform(xses_series, train_path.k, train_path.cmeans_centers)
    #     print([train_path.points[step].get_error(xses_series[train_path.input_data_index])
    #         for step in range(len(train_path.points))])
    
    
    # examine error of fcms
    input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    xses_series, ys = loadArff.load_cricket_normalized(input_file)
    xses_series_derived = [derivatives.transform(xses_series, order) for order in [0,1,2]]

    for derivative_order in [0]:
        for no_centers in [3, 4, 7, 10, 15, 20, 25, 30, 35]:
            checkpoints_train_dir = pathlib.Path(f'./checkpoints/Cricket/fcm_cmeans/{no_centers}_{0.002}_{derivative_order}/train')
            checkpoints_test_dir = pathlib.Path(f'./checkpoints/Cricket/fcm_cmeans/{no_centers}_{0.002}_{derivative_order}/test')

            train_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_train_dir)
            test_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_test_dir)
            
            steps = len(train_training_paths[0].points)

            plot_xs = [i for i in range(0, steps)]
            plot_ys = [[] for x in plot_xs]

            xses_series_transformed = xses_series_derived[derivative_order]
            if train_training_paths[0].cmeans_centers is not None:
                xses_series_transformed = cmeans.transform(xses_series_transformed, train_training_paths[0].cmeans_centers)

            print("Calculating prediction error")
            for train_path in tqdm(train_training_paths):
                xs = xses_series_transformed[train_path.input_data_index]
                for step in range(len(train_path.points)):
                    err = train_path.points[step].get_error(xs)
                    plot_ys[step].append(err)

            plot_ys_mean = [np.mean(plot_ys[step]) for step in range(steps)]
            plot_ys_sd = [np.std(plot_ys[step]) for step in range(steps)]

            color = 'tab:blue'
            fig, ax = plt.subplots()
            ax.errorbar(plot_xs, plot_ys_mean, plot_ys_sd, color=color)
            ax.set(xlabel='step (lr 0.002)', ylabel='mean prediction error',
                   title=f'Cricket, c={no_centers}, derivative {derivative_order}')
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid()

            plot_ys2 = [0 for x in plot_xs]

            print("Calculating classification error")
            for step in tqdm(range(steps)):
                train_models = [tp.points[step] for tp in train_training_paths]
                test_models = [tp.points[step] for tp in test_training_paths]
                rf_accuracy = compare_solutions(
                    train_models=train_models,
                    test_models=test_models,
                    test_xs=xses_series_transformed,
                    test_ys=None,
                    input_size=no_centers,
                    extend_size=0,
                    no_classes=12)
                plot_ys2[step] = rf_accuracy

            color = 'tab:red'
            ax2 = ax.twinx()
            ax2.set_ylabel('classification accuracy (rf)')
            ax2.plot(plot_xs, plot_ys2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.savefig(pathlib.Path(f'./plots/Cricket/pred_vs_class_by_step_{no_centers}_{0.002}_{derivative_order}.png'))
            plt.close()


    # # solution comparison for fcms
    # checkpoints_train_dir = pathlib.Path('./checkpoints/Cricket/fcm_cmeans/6_0.002_2/train')
    # checkpoints_test_dir = pathlib.Path('./checkpoints/Cricket/fcm_cmeans/6_0.002_2/test')
    # input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # train_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_train_dir)
    # test_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_test_dir)
    # for step in range(len(train_training_paths[0].points)):
    #     print(f"Step {step}")
    #     train_models = [(tp.points[step], tp.input_data_index) for tp in train_training_paths]
    #     test_models = [(tp.points[step], tp.input_data_index) for tp in test_training_paths]
    #     compare_solutions(test_models, test_models, None, None, input_size=6, extend_size=0, no_classes=12)


    # # solution comparison for mppi
    # train_input_file = pathlib.Path('./data/Cricket/CRICKET_TRAIN.arff')
    # test_input_file = pathlib.Path('./data/Cricket/CRICKET_TEST.arff')
    # train_xses_series, train_ys = loadArff.load_cricket_normalized(train_input_file)
    # test_xses_series, test_ys = loadArff.load_cricket_normalized(test_input_file)
    # train_models = []
    # input_data_index = 0
    # for xs, y in zip(train_xses_series, train_ys):
    #     mppicm = MppiCognitiveMap(6)
    #     mppicm.train(xs)
    #     mppicm.set_class(y)
    #     train_models.append((mppicm, input_data_index))
    #     input_data_index+=1
    # test_models = []
    # input_data_index = 0
    # for xs, y in zip(test_xses_series, test_ys):
    #     mppicm = MppiCognitiveMap(6)
    #     mppicm.train(xs)
    #     mppicm.set_class(y)
    #     test_models.append((mppicm, input_data_index))
    #     input_data_index+=1
    # compare_solutions(test_models, test_models, test_xses_series, test_ys, 6, 0, 12)

    # solution comparison for many derivative cmeans fcms
    # input_file = pathlib.Path('./data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.arff')
    # xses_series, ys = loadArff.load_cricket_normalized(input_file)
    # xses_series_derived = [derivatives.transform(xses_series, order) for order in [0,1,2]]
    # plot_xs, plot_ys = [], []
    # plot2_xs, plot2_ys = [], []
    # plot3_xs, plot3_ys = [], []
    # for no_centers in range(2, 38):
    #     for derivative_order in [0]:
    #         checkpoints_train_dir = pathlib.Path(f'./checkpoints/UWaveGestureLibrary/fcm_cmeans/{no_centers}_{0.002}_{derivative_order}/train')
    #         checkpoints_test_dir = pathlib.Path(f'./checkpoints/UWaveGestureLibrary/fcm_cmeans/{no_centers}_{0.002}_{derivative_order}/test')

    #         train_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_train_dir)
    #         test_training_paths = fcmCheckpoints.load_checkpoints(checkpoints_test_dir)

    #         xses_series_transformed = xses_series_derived[derivative_order]
    #         xses_series_transformed = cmeans.transform(
    #             xses_series=xses_series_transformed,
    #             centers=test_training_paths[0].cmeans_centers)

    #         xses_series_transformed = [xses_series_transformed[tp.input_data_index] for tp in test_training_paths]

    #         train_models = [tp.points[-1] for tp in train_training_paths]
    #         test_models = [tp.points[-1] for tp in test_training_paths]

    #         print(f'no_centers {no_centers}, derivative_order {derivative_order}')
    #         rf_accuracy = compare_solutions(
    #             train_models=train_models,
    #             test_models=test_models,
    #             test_xs=xses_series_transformed,
    #             test_ys=None,
    #             input_size=no_centers,
    #             extend_size=0,
    #             no_classes=12)
    #         err = sum([tp.points[-1].get_error(xs) for tp, xs in zip(train_training_paths, xses_series_transformed)])
    #         print(f'Prediction error: {err / no_centers}')
    #         plot_xs.append(err / no_centers)
    #         plot_ys.append(rf_accuracy)
    #         plot2_xs.append(no_centers)
    #         plot2_ys.append(err / no_centers)
    #         plot3_xs.append(no_centers)
    #         plot3_ys.append(rf_accuracy)

    # fig, ax = plt.subplots()
    # ax.plot(plot_xs, plot_ys, 'bo')
    # ax.set(xlabel='sum of mean fcm prediction errors', ylabel='classification accuracy (rf)',
    #    title='UWaveGestureLibrary, no derivatives')
    # ax.grid()

    # fig, ax = plt.subplots()
    # ax.plot(plot2_xs, plot2_ys, 'bo')
    # ax.set(xlabel='number of cmeans centers', ylabel='prediction error',
    #    title='no derivatives')
    # ax.grid()

    # fig, ax = plt.subplots()
    # ax.plot(plot3_xs, plot3_ys, 'bo')
    # ax.set(xlabel='number of cmeans centers', ylabel='classification accuracy (rf)',
    #    title='UWaveGestureLibrary, no derivatives')
    # ax.grid()


    plt.show()