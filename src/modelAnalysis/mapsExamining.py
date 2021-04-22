import numpy as np


def get_share_of_degenerated_weights(models, threshold):
    no_degenerated_weights = 0
    for model in models:
        no_degenerated_weights += np.sum(model.weights >= threshold)
        no_degenerated_weights += np.sum(model.weights <= -threshold)
    return no_degenerated_weights/(models[0].n*models[0].n*len(models))
