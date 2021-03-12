import numpy as np
from skfuzzy import cmeans, cmeans_predict


def find_centers_and_transform(xses_series, c, m=2.0):
    squashed_xses = []
    for xs in xses_series:
        squashed_xses.extend(xs)
    squashed_xses = np.asarray(squashed_xses).transpose()
    init_centers = np.asarray([((i+1)/(c+1))*np.ones(shape=(len(xses_series[0][0]))) for i in range(c)])
    init_cpartitioned_matrix = cmeans_predict(
        squashed_xses,
        init_centers,
        m,
        error=0.001,
        maxiter=1000)[0]
    result = cmeans(data=squashed_xses, c=c, m=m, error=0.001, maxiter=1000, init=init_cpartitioned_matrix)
    centers = result[0].tolist()
    memberships = np.asarray(result[1]).transpose()

    no_transformed_xs = 0
    transformed_xses_series = []
    for xs in xses_series:
        transformed_xses_series.append(
            memberships[no_transformed_xs:no_transformed_xs+len(xs)].tolist()
            )
        no_transformed_xs += len(xs)

    return centers, transformed_xses_series



def find_centers_in_first_and_transform_second(first_series, second_series, c):
    raise Exception("Not up-to-date, veri sori")
    squashed_xses = []
    for xs in first_series:
        squashed_xses.extend(xs)
    squashed_xses = np.asarray(squashed_xses).transpose()
    result = cmeans(data=squashed_xses, c=c, m=2.0, error=0.001, maxiter=1000)
    centers = result[0].tolist()

    return centers, transform(second_series, centers)


def transform(xses_series, centers):
    squashed_xses = []
    for xs in xses_series:
        squashed_xses.extend(xs)
    squashed_xses = np.asarray(squashed_xses).transpose()
    centers=np.asarray(centers)
    result = cmeans_predict(
        test_data=squashed_xses,
        cntr_trained=centers,
        m=2.0,
        error=0.001,
        maxiter=1000
        )
    memberships = np.asarray(result[0]).transpose()
    no_transformed_xs = 0
    transformed_xses_series = []
    for xs in xses_series:
        transformed_xses_series.append(
            memberships[no_transformed_xs:no_transformed_xs+len(xs)].tolist()
            )
        no_transformed_xs += len(xs)

    return transformed_xses_series
