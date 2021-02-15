import numpy as np
import copy


def transform(xses_series):
    transformed_xses_series = []
    velocities_series = [_velocities(np.asarray(xs)) for xs in xses_series]
    velocity_mean = np.mean(np.asarray(velocities_series))
    velocities_series = [_sigmoid(np.divide(vs, velocity_mean)) for vs in velocities_series]
    for xs, vs in zip(xses_series, velocities_series):
        distance_since_last_point = 0.0
        transformed_xs = []
        for x, v in zip(xs, vs):
            distance_since_last_point += v
            while distance_since_last_point > 1.0:
                transformed_xs.append(x)
                distance_since_last_point -= 1.0
        transformed_xses_series.append(transformed_xs)

    return transformed_xses_series

def _velocities(xs):
    return np.sqrt(np.sum(np.square(xs[:-1]-xs[1:]), axis=1))

def _sigmoid(x):
    return np.square(x)