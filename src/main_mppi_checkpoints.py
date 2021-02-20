

import pathlib

from cognitiveMaps import mppiCheckpoints
from transformingData import derivatives
from transformingData import normalizing
from transformingData import cmeans
from loadingData import loadArff


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

